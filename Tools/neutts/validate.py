"""Generate a reproducible speaker/emotion audio matrix using the exported graphs."""
import argparse
import json
import time
import unicodedata
from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tokenizers import Tokenizer


def normalize(text):
    return unicodedata.normalize("NFKC", text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})))


def prompt(tokenizer, meta, text, speaker, emotion):
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    reference = next(s for s in meta['speakers'] if s['name'] == speaker)
    if emotion == 'neutral':
        ids = encode(normalize(reference['text']) + ' ' + normalize(text))
    else:
        eid = next(e['token'] for e in meta['emotions'] if e['name'] == emotion)
        ids = encode(normalize(reference['text'])) + [eid] + encode(normalize(text))
    return [meta['textStart']] + ids + [meta['textEnd'], meta['speechStart']] + [meta['speechTokenIds'][i] for i in reference['codes']]


def generate(session, meta, ids, seed=42, maximum=700, top_p=1.0, min_p=0.0):
    if not 0 < top_p <= 1 or not 0 <= min_p <= 1:
        raise ValueError('top_p must be in (0, 1] and min_p in [0, 1].')
    cache = {f'past_key_values.{i}.{kind}': np.empty((1, meta['kvHeads'], 0, meta['headDim']), np.float32)
        for i in range(meta['layers']) for kind in ['key', 'value']}
    rng = np.random.default_rng(seed)
    previous = 0
    generated = []
    for step in range(min(maximum, meta['maxContext'] - len(ids))):
        positions = np.arange(previous, previous + len(ids), dtype=np.int64)[None, :]
        mask = np.where(np.arange(previous + len(ids))[None, :] > positions[0, :, None], -np.inf, 0).astype(np.float32)[None, None, :, :]
        outputs = session.run(None, dict(cache, input_ids=np.array([ids], np.int64), position_ids=positions, attention_mask=mask))
        logits = outputs[0].reshape(-1).astype(np.float64)
        if step < 50:
            logits[meta['speechEnd']] = -np.inf
        candidates = np.argsort(-logits, kind='stable')[:50]
        weights = np.exp(logits[candidates] - logits[candidates[0]])
        probabilities = weights / weights.sum()
        keep = (np.cumsum(probabilities) - probabilities < top_p) & (weights >= min_p * weights[0])
        candidates, weights = candidates[keep], weights[keep]
        next_id = int(rng.choice(candidates, p=weights / weights.sum()))
        if next_id == meta['speechEnd']:
            return generated, True
        generated.append(next_id)
        cache = {name.replace('present.', 'past_key_values.'): value for name, value in zip([o.name for o in session.get_outputs()][1:], outputs[1:])}
        previous += len(ids)
        ids = [next_id]
    return generated, False


def main(args):
    root = args.artifacts
    meta = json.loads((root / 'neutts.json').read_text())
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(root / f'onnx/backbone_{args.profile}.onnx'), options, providers=['CPUExecutionProvider'])
    codec = ort.InferenceSession(str(root / 'onnx/codec_decoder.onnx'), options, providers=['CPUExecutionProvider'])
    code_map = {tid: code for code, tid in enumerate(meta['speechTokenIds'])}
    samples = root / 'samples' / args.profile
    samples.mkdir(parents=True, exist_ok=True)
    results = []
    fixtures = []
    for speaker in meta['speakers']:
        for emotion in meta['emotions']:
            text = "I can't believe it's finally here! We have been waiting for this moment."
            ids = prompt(tokenizer, meta, text, speaker['name'], emotion['name'])
            fixtures.append(dict(text=text, speaker=speaker['name'], emotion=emotion['name'], ids=ids))
            started = time.perf_counter()
            generated, stopped = generate(session, meta, ids, top_p=args.top_p, min_p=args.min_p)
            backbone_seconds = time.perf_counter() - started
            codes = [code_map[i] for i in generated if i in code_map]
            audio = codec.run(None, {'codes': np.array([[codes]], np.int32)})[0].reshape(-1)
            assert len(audio) and np.isfinite(audio).all() and np.max(np.abs(audio)) > 0.001
            name = f"{speaker['name']}-{emotion['name']}"
            sf.write(samples / f'{name}.wav', audio, 24000, subtype='PCM_16')
            record = dict(name=name, text=text, tokens=len(generated), speech_tokens=len(codes), stopped=stopped,
                backbone_seconds=backbone_seconds, total_seconds=time.perf_counter()-started,
                duration=len(audio)/24000, peak=float(np.max(np.abs(audio))), rms=float(np.sqrt(np.mean(audio**2))))
            results.append(record)
            (root / f'validation-{args.profile}.json').write_text(json.dumps(results, indent=2))
            print(json.dumps(record), flush=True)
            if args.smoke:
                break
        if args.smoke:
            break
    # Ground-truth prompt fixtures are written by reference_parity.py, not this implementation.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('--profile', choices=['fp32', 'fp16', 'int8', 'int4'], default='fp32')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--min-p', type=float, default=0.0)
    main(parser.parse_args())
