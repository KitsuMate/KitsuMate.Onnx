"""Compare prepared NeuTTS logits and KV outputs on identical reference continuations.

Timings here include CPU cache copies and are not an end-to-end RTF benchmark.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from compare_profiles import empty, feeds
from validate import prompt


def validate(args):
    metadata = json.loads((args.artifacts/'neutts.json').read_text())
    tokenizer = Tokenizer.from_file(str(args.artifacts/'tokenizer.json'))
    texts = ['Hello, this is a local voice test.',
             'At three forty-five, please bring twelve blue folders to room seven.',
             "I can't believe it's finally here! We have been waiting for this moment.",
             'The train was late, so we walked along the river and watched the evening lights come on.']
    cases = [(speaker['name'], texts[i], 'happy' if i == 2 else 'neutral')
             for i, speaker in enumerate(metadata['speakers'])]
    references, report = [], []
    for prepared, path in [(False, args.source), (True, args.prepared)]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        if prepared and args.webgpu:
            import onnxruntime_ep_webgpu as webgpu
            ort.register_execution_provider_library('validate_neutts_backbone', webgpu.get_library_path())
            devices = [d for d in ort.get_ep_devices() if d.ep_name == webgpu.get_ep_name()]
            options.add_provider_for_devices([devices[args.device]], {})
            options.enable_profiling = True
            options.profile_file_prefix = str(args.report.with_suffix(''))+'-profile'
        session = ort.InferenceSession(str(path), options)
        for case, (speaker, text, emotion) in enumerate(cases):
            ids = prompt(tokenizer, metadata, text, speaker, emotion)
            previous, cache, trace = 0, empty(metadata), []
            worst_logits, worst_cache, overlap = 0, 0, []
            for step in range(args.steps+1):
                outputs = session.run(None, feeds(metadata, ids, previous, cache))
                logits = outputs[0].reshape(-1)
                if not all(np.isfinite(o).all() for o in outputs):
                    raise AssertionError(f'Nonfinite output: {speaker}, step {step}')
                if not prepared:
                    # Cache reference summaries avoid retaining several GB per case.
                    trace.append((logits.copy(), [o[:, :, -1:, :].copy() for o in outputs[1:]]))
                reference = references[case][step] if prepared else trace[-1]
                if prepared:
                    worst_logits = max(worst_logits, float(np.max(np.abs(logits-reference[0]))))
                    worst_cache = max(worst_cache, max(float(np.max(np.abs(o[:, :, -1:, :]-r)))
                        for o, r in zip(outputs[1:], reference[1])))
                    overlap.append(len(set(np.argsort(logits)[-50:]) & set(np.argsort(reference[0])[-50:]))/50)
                previous += len(ids)
                ids = [int(np.argmax(reference[0]))]
                cache = dict(zip([o.name.replace('present.', 'past_key_values.') for o in session.get_outputs()[1:]], outputs[1:]))
            if prepared:
                result = dict(speaker=speaker, steps=args.steps+1, max_logit_error=worst_logits,
                              max_new_cache_error=worst_cache, top50_overlap=float(np.mean(overlap)))
                assert worst_logits < 0.001 and worst_cache < 0.001 and min(overlap) >= 0.98, result
                report.append(result)
                print(json.dumps(result), flush=True)
            else:
                references.append(trace)
        if prepared and args.webgpu:
            events = json.loads(Path(session.end_profiling()).read_text())
            counts = collections.Counter((e['args'].get('provider'), e['args'].get('op_name'))
                for e in events if e.get('args', {}).get('provider'))
            assert counts[('WebGpuExecutionProvider', 'GroupQueryAttention')] > 0, counts
            assert counts[('CPUExecutionProvider', 'GroupQueryAttention')] == 0, counts
            report.append(dict(operator_executions={str(k): v for k, v in counts.items()}))
        del session, cache, outputs
    args.report.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--webgpu', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--report', required=True, type=Path)
    validate(parser.parse_args())
