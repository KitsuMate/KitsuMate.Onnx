"""Compare profiles on identical FP32-generated continuations, without concurrent workloads."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from validate import prompt


def feeds(meta, ids, previous, cache):
    positions = np.arange(previous, previous + len(ids), dtype=np.int64)[None]
    mask = np.where(np.arange(previous + len(ids))[None] > positions[0, :, None],
                    -np.inf, 0).astype(np.float32)[None, None]
    return dict(cache, input_ids=np.array([ids], np.int64), position_ids=positions, attention_mask=mask)


def empty(meta):
    return {f'past_key_values.{i}.{kind}': np.empty((1, meta['kvHeads'], 0, meta['headDim']), np.float32)
            for i in range(meta['layers']) for kind in ['key', 'value']}


def main(args):
    root = args.artifacts
    meta = json.loads((root / 'neutts.json').read_text())
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    corpus = [("Hello, this is a local voice test.", 'emily', 'neutral'),
              ("I can't believe it's finally here! We have been waiting for this moment.", 'sophie', 'happy'),
              ("At three forty-five, please bring twelve blue folders to room seven.", 'paul', 'neutral')]
    references = []
    results = []
    for profile in args.profiles:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.enable_profiling = args.provider == 'CUDAExecutionProvider'
        options.profile_file_prefix = str(root / ('comparison-trace-' + profile))
        if options.enable_profiling:
            ort.preload_dlls()
        path = root / f'onnx/backbone_{profile}.onnx'
        session = ort.InferenceSession(str(path), options, providers=[args.provider])
        names = [o.name.replace('present.', 'past_key_values.') for o in session.get_outputs()[1:]]
        prefill, decode, errors, overlaps, agreements = [], [], [], [], []
        # Warm up kernels before timing.
        session.run(None, feeds(meta, [100, 200], 0, empty(meta)))
        for case, (text, speaker, emotion) in enumerate(corpus):
            ids = prompt(tokenizer, meta, text, speaker, emotion)
            cache, previous, trace = empty(meta), 0, []
            for step in range(args.steps + 1):
                started = time.perf_counter()
                outputs = session.run(None, feeds(meta, ids, previous, cache))
                elapsed = time.perf_counter() - started
                (prefill if step == 0 else decode).append(elapsed)
                logits = outputs[0].reshape(-1)
                assert np.isfinite(logits).all(), (profile, case, step)
                if profile == 'fp32':
                    trace.append(logits.copy())
                    reference = logits
                else:
                    reference = references[case][step]
                    errors.append(float(np.max(np.abs(logits - reference))))
                    agreements.append(int(np.argmax(logits) == np.argmax(reference)))
                    overlaps.append(len(set(np.argpartition(logits, -50)[-50:]) &
                                        set(np.argpartition(reference, -50)[-50:])) / 50)
                previous += len(ids)
                ids = [int(np.argmax(reference))]
                cache = dict(zip(names, outputs[1:]))
            if profile == 'fp32':
                references.append(trace)
        report = dict(profile=profile, provider=args.provider, threads=4,
                      bytes=path.stat().st_size + sum(p.stat().st_size for p in path.parent.glob(path.name + '.*')),
                      prefill_median_seconds=float(np.median(prefill)), decode_tokens_per_second=len(decode)/sum(decode) if decode else None,
                      teacher_forced_steps=len(agreements), max_logit_error=max(errors, default=0),
                      argmax_agreement=float(np.mean(agreements)) if agreements else 1,
                      top50_overlap=float(np.mean(overlaps)) if overlaps else 1)
        if options.enable_profiling:
            events = json.loads(Path(session.end_profiling()).read_text())
            counts = {}
            for event in events:
                provider = event.get('args', {}).get('provider')
                if provider:
                    counts[provider] = counts.get(provider, 0) + 1
            assert counts.get(args.provider, 0), counts
            report['provider_node_executions'] = counts
        results.append(report)
        (root / ('profile-comparison-' + args.provider + '.json')).write_text(json.dumps(results, indent=2))
        print(json.dumps(report), flush=True)
        del session, outputs, cache
        gc.collect()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('artifacts', type=Path)
    p.add_argument('--profiles', nargs='+', default=['fp32', 'fp16', 'int8', 'int4'])
    p.add_argument('--provider', default='CPUExecutionProvider')
    p.add_argument('--steps', type=int, default=50)
    args = p.parse_args()
    assert args.profiles[0] == 'fp32', 'FP32 must run first to supply reference tokens.'
    main(args)
