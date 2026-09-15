"""Check prompts and cached logits against the pinned upstream implementation."""
import argparse
import ast
import json
import types
import unicodedata
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import Tokenizer
from validate import prompt


def main(args):
    torch.set_num_threads(2)
    root = args.artifacts
    meta = json.loads((root / 'neutts.json').read_text())
    hf_tokenizer = AutoTokenizer.from_pretrained(args.source)
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    tree = ast.parse((args.upstream / 'neutts/neutts.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NeuTTS')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_apply_chat_template')
    normalize = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_normalize_text')
    env = {'unicodedata': unicodedata, '_QUOTE_MAP': str.maketrans({'‘': "'", '’': "'", '“': '"', '”': '"'})}
    exec(compile(ast.Module(body=[normalize, method], type_ignores=[]), '<pinned upstream prompt>', 'exec'), env)
    owner = types.SimpleNamespace(tokenizer=hf_tokenizer, input_format='BPE')
    cases = []
    for speaker in meta['speakers']:
        for emotion in meta['emotions']:
            for text in ["I can't believe it's finally here!", '“Hello,” she said. Café № １２ — it’s 3:45.']:
                expected = env['_apply_chat_template'](owner, speaker['codes'], speaker['text'], text,
                    None if emotion['name'] == 'neutral' else emotion['name'])
                actual = prompt(tokenizer, meta, text, speaker['name'], emotion['name'])
                assert actual == expected, (speaker['name'], emotion['name'], text)
                cases.append(dict(text=text, speaker=speaker['name'], emotion=emotion['name'], ids=expected))
    (root / 'prompt-fixtures.json').write_text(json.dumps({'cases': cases}, indent=2), encoding='utf-8')
    model = AutoModelForCausalLM.from_pretrained(args.source, dtype=torch.float32, attn_implementation='eager').eval()
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(root / 'onnx/backbone_fp32.onnx'), options, providers=['CPUExecutionProvider'])
    ids = cases[0]['ids']
    cache = {f'past_key_values.{i}.{k}': np.empty((1,meta['kvHeads'],0,meta['headDim']),np.float32)
        for i in range(meta['layers']) for k in ['key','value']}
    native_cache = None
    previous = 0
    errors = []
    for step in range(12):
        values = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            reference = model(input_ids=values, past_key_values=native_cache, use_cache=True)
        native_cache = reference.past_key_values
        positions = np.arange(previous, previous+len(ids),dtype=np.int64)[None,:]
        mask = np.where(np.arange(previous+len(ids))[None,:] > positions[0,:,None],-np.inf,0).astype(np.float32)[None,None,:,:]
        actual = session.run(None,dict(cache,input_ids=values.numpy(),position_ids=positions,attention_mask=mask))
        expected = reference.logits[:,-1:,:].numpy()
        np.testing.assert_allclose(actual[0], expected, atol=0.002, rtol=0.002)
        errors.append(float(np.max(np.abs(actual[0]-expected))))
        for index, layer in enumerate(native_cache.layers):
            np.testing.assert_allclose(actual[1+2*index],layer.keys.numpy(),atol=0.002,rtol=0.002)
            np.testing.assert_allclose(actual[2+2*index],layer.values.numpy(),atol=0.002,rtol=0.002)
        previous += len(ids)
        ids = [int(np.argmax(expected))]
        cache = {name.replace('present.','past_key_values.'): value for name,value in zip([o.name for o in session.get_outputs()][1:],actual[1:])}
    (root/'reference-validation.json').write_text(json.dumps({'prompt_cases':len(cases),'cached_steps':len(errors),'max_logit_errors':errors},indent=2))
    print('Upstream prompt and cached-logit parity passed.',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ['source','upstream','artifacts']: p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
