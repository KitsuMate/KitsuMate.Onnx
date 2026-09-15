"""Validate FP32 on CUDA and record actual node placement from ORT profiling."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from tokenizers import Tokenizer
from validate import generate, prompt

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('artifacts',type=Path);args=p.parse_args()
    ort.preload_dlls()
    assert 'CUDAExecutionProvider' in ort.get_available_providers()
    meta=json.loads((args.artifacts/'neutts.json').read_text())
    tokenizer=Tokenizer.from_file(str(args.artifacts/'tokenizer.json'))
    options=ort.SessionOptions(); options.intra_op_num_threads=2; options.enable_profiling=True
    options.profile_file_prefix=str(args.artifacts/'cuda-profile')
    session=ort.InferenceSession(str(args.artifacts/'onnx/backbone_fp32.onnx'),options,providers=['CUDAExecutionProvider'])
    codec=ort.InferenceSession(str(args.artifacts/'onnx/codec_decoder.onnx'),options,providers=['CUDAExecutionProvider'])
    assert session.get_providers()[0]=='CUDAExecutionProvider'
    ids=prompt(tokenizer,meta,'Hello, this is a local voice test.','emily','neutral')
    started=time.perf_counter(); tokens,stopped=generate(session,meta,ids)
    mapping={tid:i for i,tid in enumerate(meta['speechTokenIds'])}
    codes=[mapping[t] for t in tokens if t in mapping]
    audio=codec.run(None,{'codes':np.array([[codes]],np.int32)})[0]
    elapsed=time.perf_counter()-started
    assert stopped and np.isfinite(audio).all() and audio.size>24000
    providers=[]
    for current in [session,codec]:
        path=current.end_profiling()
        events=json.loads(Path(path).read_text())
        counts={}
        for event in events:
            provider=event.get('args',{}).get('provider')
            if provider: counts[provider]=counts.get(provider,0)+1
        assert counts.get('CUDAExecutionProvider',0)>0
        providers.append(counts)
    report=dict(gpu=torch.cuda.get_device_name(),duration=audio.size/24000,seconds=elapsed,
        tokens=len(tokens),provider_node_executions=providers)
    (args.artifacts/'cuda-validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
