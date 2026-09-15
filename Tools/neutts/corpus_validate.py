"""Check short, numeric and longer utterances on both profiles."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tokenizers import Tokenizer
from validate import generate, prompt

TEXTS = [
    ('short', 'Hello!', 'Hello'),
    ('numbers', 'At 3:45, please bring 12 blue folders to room 7.', 'At three forty five please bring twelve blue folders to room seven'),
    ('long', 'The train was late, so we walked along the river and watched the evening lights come on. '
        'When we reached the old bridge, we stopped for a moment, took a photograph, and decided to return the next day.', None),
]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('artifacts',type=Path)
    parser.add_argument('--numbers-as-words',action='store_true');args=parser.parse_args()
    if args.numbers_as_words:
        TEXTS=[('numbers-as-words','At three forty-five, please bring twelve blue folders to room seven.',None)]
    root=args.artifacts;meta=json.loads((root/'neutts.json').read_text());tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    codec=ort.InferenceSession(str(root/'onnx/codec_decoder.onnx'),options,providers=['CPUExecutionProvider'])
    mapping={tid:i for i,tid in enumerate(meta['speechTokenIds'])}
    results=[]
    for profile in ['fp32','int4']:
        session=ort.InferenceSession(str(root/f'onnx/backbone_{profile}.onnx'),options,providers=['CPUExecutionProvider'])
        for name,text,expected in TEXTS:
            start=time.perf_counter()
            tokens,stopped=generate(session,meta,prompt(tokenizer,meta,text,'emily','neutral'),maximum=1100)
            codes=[mapping[t] for t in tokens if t in mapping]
            audio=codec.run(None,{'codes':np.array([[codes]],np.int32)})[0].reshape(-1)
            assert stopped and audio.size>0 and np.isfinite(audio).all()
            target=root/'samples'/profile/f'corpus-{name}.wav';sf.write(target,audio,24000)
            record=dict(profile=profile,name=f'corpus-{name}',text=text,expected=expected or text,stopped=stopped,
                tokens=len(tokens),seconds=time.perf_counter()-start,duration=audio.size/24000)
            results.append(record);print(json.dumps(record),flush=True)
        del session
    (root/('numbers-as-words-validation.json' if args.numbers_as_words else 'corpus-validation.json')).write_text(json.dumps(results,indent=2))
