"""Build model-card, schemas, attribution and checksums for validated artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path
import onnx


def main(root):
    reference=json.loads((root/'reference-validation.json').read_text())
    assert reference['prompt_cases']==56
    profiles = [p.stem.removeprefix('backbone_') for p in (root/'onnx').glob('backbone_*.onnx')]
    for profile in profiles:
        records=json.loads((root/f'validation-{profile}.json').read_text())
        assert len(records)==28 and all(r['stopped'] for r in records)
    asr=json.loads((root/'asr-validation.json').read_text())
    for profile in profiles:
        records = [r for r in asr if r['profile'] == profile]
        assert len(records)==28 and max(r['wer'] for r in records)<=0.15, profile
    meta=json.loads((root/'neutts.json').read_text())
    schemas={}
    for path in (root/'onnx').glob('*.onnx'):
        graph=onnx.load(str(path),load_external_data=False)
        if path.name == 'codec_decoder.onnx' and any(n.op_type in ('ConvTranspose', 'SplitToSequence', 'SequenceAt') for n in graph.graph.node):
            raise ValueError('Run prepare_codec.py before publishing the codec.')
        if path.name.startswith('backbone'):
            if not graph.doc_string:
                graph.doc_string='Modified by KitsuMate: NeuTTS-2E ONNX export with explicit KV cache and precision conversion named in the file. Original model by Neuphonic; see LICENSE and ATTRIBUTION.md.'
            onnx.save(graph,str(path))
        def tensor(value):
            t=value.type.tensor_type
            return dict(name=value.name,type=onnx.TensorProto.DataType.Name(t.elem_type),
                shape=[d.dim_param or d.dim_value for d in t.shape.dim])
        schemas[path.name]={'inputs':[tensor(i) for i in graph.graph.input],'outputs':[tensor(o) for o in graph.graph.output]}
    (root/'graph-schemas.json').write_text(json.dumps(schemas,indent=2))
    versions={name:importlib.metadata.version(name) for name in ['torch','transformers','onnx','onnxruntime','numpy','tokenizers','huggingface_hub']}
    versions['python']=platform.python_version()
    (root/'export-environment.json').write_text(json.dumps(versions,indent=2))
    (root/'ATTRIBUTION.md').write_text(f'''# Attribution and modifications

NeuTTS-2E was created by Neuphonic. Original checkpoint: neuphonic/neutts-2e at `{meta['sourceRevision']}`.
Backbone weights remain under the NeuTTS Open License v1.0, reproduced in LICENSE.
KitsuMate converted the backbone to ONNX, exposed dynamic sequence/cache inputs, returns last-position logits,
and created a symmetric INT4 weight-only profile (block size 128; MatMul and embedding Gather weights).
These changes are identified in each backbone graph's documentation string.
The FP16 profile stores matrix weights in float16 and casts them to float32 for computation.
The INT8 profile uses dynamic QInt8 MatMul and quantized Gather weights.
Our comparison was informed by Danny-Dasilva/neutts-2e-onnx at
`e3ea12b860d75acf6903ddd4fbb7c72c0ed4eae3`; these artifacts are converted from our own
validated FP32 graph, not copied from that repository. Our graph interface remains unchanged.

The NeuCodec decoder derives from neuphonic/neucodec-onnx-decoder at `{meta['codecRevision']}`.
KitsuMate replaces its iSTFT ConvTranspose with an equivalent overlap Conv and sample
rearrangement for WebGPU correctness, and sequence indexing with tensor Gather to avoid
CPU/GPU transfers, using `tools/prepare_codec.py`. The codec remains
distributed under Apache-2.0 (CODEC_LICENSE). Speaker reference codes/transcripts come from
https://github.com/neuphonic/neutts at `{meta['upstreamRevision']}` (Apache-2.0).
KitsuMate converted these references from PyTorch tensors/text files to neutts.json.
The original authors retain their copyrights and attribution. This conversion is not an endorsement by Neuphonic.
''')
    (root/'README.md').write_text('''---
license: other
license_name: neutts-open-license-1.0
license_link: LICENSE
language:
- en
pipeline_tag: text-to-speech
library_name: onnx
base_model: neuphonic/neutts-2e
tags:
- onnx
- text-to-speech
- emotional-tts
---

# NeuTTS-2E ONNX

KitsuMate ONNX conversion of Neuphonic's English fixed-speaker emotional TTS model.
Preview release: numerical and automated speech checks passed. Full Unity regression testing
and subjective listening review are still incomplete; see unity-validation.json.
Speakers: `emily`, `paul`, `sophie`, `steven`.
Emotions: `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`.

## Profiles

| Profile | Backbone | Shared decoder | Intended execution |
|---|---|---|---|
| Portable FP32 | onnx/backbone_fp32.onnx | onnx/codec_decoder.onnx | ONNX Runtime CPU or CUDA |
| FP16 weight storage | onnx/backbone_fp16.onnx | onnx/codec_decoder.onnx | FP32 computation; smaller download |
| CPU dynamic INT8 | onnx/backbone_int8.onnx | onnx/codec_decoder.onnx | ONNX Runtime CPU |
| CPU compact INT4 | onnx/backbone_int4.onnx + .data companion | onnx/codec_decoder.onnx | ONNX Runtime CPU |

The INT4 graph uses ONNX Runtime contrib operators and opset 21. FP32 uses standard opset 17.
FP16 means half-precision matrix weight storage here, with FP32 arithmetic and tensor interfaces.
It is not a full FP16-compute graph and does not promise lower runtime memory use.
See profile-comparison-CPUExecutionProvider.json for matched-input timing and numerical comparisons.
Download tokenizer.json and neutts.json for both profiles. Do not omit the INT4 external-data companion.
The backbone accepts an empty cache for prefill and populated caches for later tokens; only final-position logits are returned.
See graph-schemas.json for exact input/output names, types and dimensions.
The codec accepts int32 `[1,1,N]` speech codes and returns float32 mono 24 kHz audio.
Its output has `480*N-480` samples. The context budget is 2048 including reference, text and generated tokens.

## Usage

Use the NeuTTS engine/model set in KitsuMate ONNX TTS and select this repository in its downloader.
For standalone Python usage, see `tools/validate.py`: build the prompt, generate speech tokens with cached inference,
then decode them using NeuCodec. Install dependencies from tools/requirements.txt.
All reference preparation is offline; inference needs neither PyTorch nor a phonemizer in Unity.
The default speaker is emily and emotion is neutral. Sampling uses temperature 1 and top-k 50.
Unity also exposes optional TopP and MinP. Set 0.95 and 0.05 to try the community sampler;
defaults remain 1 and 0 to preserve upstream PyTorch sampling behavior.
Set a sufficient output token budget (for example 700); 50 tokens represent approximately one second.

## Validation and limitations

reference-validation.json records exact prompt parity for 56 cases and FP32 prefill/cached-logit parity.
validation-<profile>.json covers all 28 speaker/emotion combinations for each profile.
asr-validation.json records independent Whisper-base transcripts and word error rates.
cuda-validation.json records actual CUDA/CPU node placements and measured synthesis time.
unity-validation.json records the completed Unity checks and remaining validation gaps.
Samples are provided in samples/. Numerical/ASR checks do not establish subjective voice or emotion quality;
manual listening review remains outstanding. No real-time performance claim is made.
CPU matrix timings include concurrent validation workloads and are not isolated benchmarks.
CUDA comparison timings also include profiling and concurrent validation; use them as execution checks.
Some raw codec peaks exceed +/-1; normalize or limit audio before integer PCM playback/export to avoid clipping.
The supplied sample WAVs use integer PCM. The runtime returns the raw float waveform.

This implementation does not apply the optional Perth watermark. Outputs are not claimed to be watermarked.
Custom cloning, streaming, Air and Nano variants are not supported by this release.
Literal digits and times are unreliable in the upstream checkpoint and both conversions.
Spell numbers out in words before synthesis (for example, "three forty-five" instead of "3:45").
The corpus and source-checkpoint reports document this limitation; no automatic number expansion is applied.

## License

The backbone retains the NeuTTS Open License v1.0. Redistribution is subject to its license/notice conditions.
The commercial-use restriction and USD 5 million annual-revenue threshold remain applicable; see LICENSE for exact terms.
The codec and upstream reference assets are Apache-2.0; see CODEC_LICENSE and ATTRIBUTION.md.
''',encoding='utf-8')
    comparison = root/'profile-comparison-CPUExecutionProvider.json'
    if comparison.exists():
        records = json.loads(comparison.read_text())
        with (root/'README.md').open('a', encoding='utf-8') as card:
            card.write('\n## Local CPU comparison\n\nSame three prompts, 153 prefill/cached steps per profile, four ORT threads.\n'
                       'Timing excludes sampling, codec decoding and session creation; it is not end-to-end speed.\n\n'
                       '| Profile | Backbone MB | Decode tokens/s | Argmax agreement | Top-50 overlap |\n'
                       '|---|---:|---:|---:|---:|\n')
            for record in records:
                card.write(f"| {record['profile']} | {record['bytes']/1e6:.1f} | {record['decode_tokens_per_second']:.1f} | "
                           f"{record['argmax_agreement']:.1%} | {record['top50_overlap']:.1%} |\n")
            card.write('\nFP32 remains the reference. FP16 saves download space with near-reference outputs. '
                       'INT8 is smallest and has better token agreement than INT4; INT4 remains useful for its '
                       'measured decode-speed advantage. Quantized outputs differ and require listening review.\n')
    tools=root/'tools';tools.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*'):
        if source.suffix in ['.py','.txt','.md']: shutil.copy2(source,tools/source.name)
    files=[p for p in root.rglob('*') if p.is_file() and (p.suffix in ['.onnx','.data','.wav'] or
        p.parent==tools or p.name in ['LICENSE','CODEC_LICENSE','ATTRIBUTION.md','README.md','neutts.json','tokenizer.json','tokenizer_config.json',
        'config.json','generation_config.json','export-validation.json','reference-validation.json','graph-schemas.json',
        'export-environment.json','validation-fp32.json','validation-int4.json','asr-validation.json','cuda-validation.json','prompt-fixtures.json',
        'corpus-validation.json','corpus-asr-validation.json','source-numbers-validation.json','numbers-as-words-validation.json',
        'numbers-as-words-asr.json','unity-validation.json','clean-download-validation.json',
        'validation-fp16.json','validation-int8.json','profile-comparison-CPUExecutionProvider.json',
        'profile-comparison-CUDAExecutionProvider.json'])]
    checksums={p.relative_to(root).as_posix():hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files}
    (root/'checksums.json').write_text(json.dumps(checksums,indent=2))
    print('Prepared',len(files),'files for publication.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifacts',type=Path);main(p.parse_args().artifacts)
