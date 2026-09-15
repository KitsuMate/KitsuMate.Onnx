"""Export the pinned NeuTTS-2E checkpoint. Runtime consumers need no Python.

Usage: python export.py --source PATH --upstream PATH --output PATH
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

SOURCE_REVISION = "e24ca17d47b8cdd1cdab45792013b5a81547d1a5"
CODEC_REVISION = "ac01424f74ea25b245132252168e2191b3d859fa"
UPSTREAM_REVISION = "ac69851f28fc63a487917e7c2e27f0d75c759cba"
SPEAKERS = ["emily", "paul", "sophie", "steven"]
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


class Backbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, *past):
        cache = DynamicCache(list(zip(past[::2], past[1::2])), config=self.model.config)
        result = self.model.model(input_ids=input_ids,
            attention_mask={"full_attention": attention_mask}, position_ids=position_ids,
            past_key_values=cache, use_cache=True)
        logits = self.model.lm_head(result.last_hidden_state[:, -1:, :])
        return (logits, *[t for layer in cache.layers for t in (layer.keys, layer.values)])


def export(args):
    torch.set_num_threads(4)
    source, upstream, output = map(Path, (args.source, args.upstream, args.output))
    output.mkdir(parents=True, exist_ok=True)
    graphs = output / "onnx"
    graphs.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32,
        attn_implementation="eager").eval()
    config = model.config
    wrapper = Backbone(model).eval()
    kv_names = [f"past_key_values.{i}.{kind}" for i in range(config.num_hidden_layers) for kind in ("key", "value")]
    outputs = ["logits"] + [n.replace("past_key_values", "present") for n in kv_names]
    inputs = ["input_ids", "attention_mask", "position_ids"] + kv_names
    dynamic = {"input_ids": {1: "sequence"}, "attention_mask": {2: "sequence", 3: "total_sequence"},
        "position_ids": {1: "sequence"}}
    dynamic.update({n: {2: "past_sequence"} for n in kv_names})
    dynamic.update({n: {2: "total_sequence"} for n in outputs[1:]})
    ids = torch.tensor([[100, 200, 300]], dtype=torch.long)
    past = tuple(torch.zeros(1, config.num_key_value_heads, 2, config.head_dim) for _ in kv_names)
    mask = torch.zeros(1, 1, 3, 5).masked_fill(torch.arange(5)[None, :] > torch.arange(2, 5)[:, None], float("-inf"))
    example = (ids, mask, torch.arange(2, 5)[None, :], *past)
    path = graphs / "backbone_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(wrapper, example, str(path), input_names=inputs, output_names=outputs,
            dynamic_axes=dynamic, opset_version=17, dynamo=False)
    # One graph accepts zero-length cache for prefill and populated cache for decoding.
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    errors = []
    for length, previous in [(5, 0), (1, 5), (3, 7)]:
        ids = torch.arange(100, 100 + length)[None, :]
        cache = tuple(torch.zeros(1, config.num_key_value_heads, previous, config.head_dim) for _ in kv_names)
        positions = torch.arange(previous, previous + length)[None, :]
        mask = torch.zeros(1, 1, length, previous + length).masked_fill(
            torch.arange(previous + length)[None, :] > positions[0, :, None], float("-inf"))
        values = (ids, mask, positions, *cache)
        with torch.no_grad():
            expected = wrapper(*values)
        actual = session.run(None, {n: t.numpy() for n, t in zip(inputs, values)})
        for reference, result in zip(expected, actual):
            np.testing.assert_allclose(result, reference.numpy(), atol=2e-4, rtol=2e-4)
        errors.append({"sequence": length, "past": previous,
            "max_logit_error": float(np.max(np.abs(actual[0] - expected[0].numpy())))})
    (output / "export-validation.json").write_text(json.dumps(errors, indent=2))
    for name in ["tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json", "LICENSE"]:
        shutil.copy2(source / name, output / name)
    speakers = [{"name": s, "text": (upstream / f"samples/{s}.txt").read_text().strip(),
        "codes": torch.load(upstream / f"samples/{s}.pt", weights_only=True).flatten().tolist()} for s in SPEAKERS]
    speech_ids = [tokenizer.convert_tokens_to_ids(f"<|speech_{i}|>") for i in range(65536)]
    metadata = {"family": "neutts-2e", "sampleRate": 24000, "maxContext": 2048,
        "layers": config.num_hidden_layers, "kvHeads": config.num_key_value_heads, "headDim": config.head_dim,
        "speechTokenIds": speech_ids, "speechStart": tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>"),
        "speechEnd": tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>"),
        "textStart": tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>"),
        "textEnd": tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>"),
        "emotions": [{"name": e, "token": tokenizer.convert_tokens_to_ids(f"<|{e.upper()}|>") if e != "neutral" else -1} for e in EMOTIONS],
        "speakers": speakers, "sourceRevision": SOURCE_REVISION, "codecRevision": CODEC_REVISION,
        "upstreamRevision": UPSTREAM_REVISION}
    (output / "neutts.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("FP32 export and tensor parity passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True)
    export(parser.parse_args())
