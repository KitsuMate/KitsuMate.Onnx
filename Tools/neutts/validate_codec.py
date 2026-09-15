"""Compare a prepared codec against the original CPU graph using speaker references."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main(args):
    device = None
    if args.webgpu:
        import onnxruntime_ep_webgpu as webgpu
        ort.register_execution_provider_library('neutts_codec_validation', webgpu.get_library_path())
        devices = [d for d in ort.get_ep_devices() if d.ep_name == webgpu.get_ep_name()]
        if not devices:
            raise RuntimeError('No WebGPU device is available.')
        device = devices[args.device]

    def session(path, gpu=False):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        if gpu:
            options.add_provider_for_devices([device], {})
        return ort.InferenceSession(str(path), options)

    reference = session(args.source)
    prepared = session(args.prepared)
    gpu = session(args.prepared, True) if args.webgpu else None
    speakers = json.loads(args.metadata.read_text())['speakers']
    records = []
    for speaker in speakers:
        for length in sorted({2, min(64, len(speaker['codes'])), len(speaker['codes'])}):
            inputs = {'codes': np.array([[speaker['codes'][:length]]], np.int32)}
            expected = reference.run(None, inputs)[0]
            for provider, current in [('cpu', prepared), ('webgpu', gpu)]:
                if current is None:
                    continue
                start = time.perf_counter()
                actual = current.run(None, inputs)[0]
                elapsed = time.perf_counter() - start
                error = actual.astype(np.float64) - expected
                record = dict(speaker=speaker['name'], codes=length, provider=provider,
                              samples=actual.size, seconds=elapsed,
                              max_abs=float(np.max(np.abs(error))),
                              relative_l2=float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-20)))
                records.append(record)
                print(json.dumps(record), flush=True)
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps({'onnxruntime': ort.__version__, 'cases': records}, indent=2))
                if provider == 'cpu':
                    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
                else:
                    # Pointwise relative error is ill-conditioned at audio zero crossings.
                    # Require error energy below -60 dB relative to the signal, plus a
                    # peak error below 0.001 full scale. Keep both measured values above.
                    assert np.isfinite(actual).all()
                    assert record['max_abs'] <= 1e-3 and record['relative_l2'] <= 1e-3, record
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'onnxruntime': ort.__version__, 'cases': records}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('--webgpu', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='Index among WebGPU devices.')
    parser.add_argument('--report', type=Path, required=True)
    main(parser.parse_args())
