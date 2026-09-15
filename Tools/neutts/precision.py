"""Create FP16 or dynamic INT8 with the existing KitsuMate graph interface."""
import argparse
from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def convert(root, profile):
    source = root / 'onnx/backbone_fp32.onnx'
    target = root / f'onnx/backbone_{profile}.onnx'
    if profile == 'fp16':
        graph = onnx.load(source)
        # Store large weights in FP16; retain FP32 arithmetic for numerical parity.
        # This also avoids requiring float16 tensor handling in Unity's public API.
        casts = []
        for weight in graph.graph.initializer:
            if weight.data_type != onnx.TensorProto.FLOAT or len(weight.dims) < 2:
                continue
            original = weight.name
            packed = original + '_fp16_storage'
            values = onnx.numpy_helper.to_array(weight).astype('float16')
            weight.CopyFrom(onnx.numpy_helper.from_array(values, packed))
            casts.append(onnx.helper.make_node('Cast', [packed], [original],
                name=packed + '_cast', to=onnx.TensorProto.FLOAT))
        nodes = list(graph.graph.node)
        del graph.graph.node[:]
        graph.graph.node.extend(casts + nodes)
        graph.doc_string = 'Modified by KitsuMate: FP16 weight storage with FP32 computation and interfaces. See LICENSE and ATTRIBUTION.md.'
        onnx.save(graph, target)
    else:
        quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8,
            op_types_to_quantize=['MatMul', 'Gather'], extra_options={'MatMulConstBOnly': True})
        graph = onnx.load(target)
        graph.doc_string = 'Modified by KitsuMate: dynamic QInt8 MatMul and quantized Gather weights, with FP32 interfaces. See LICENSE and ATTRIBUTION.md.'
        onnx.save(graph, target)
    print(target, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('--profile', choices=['fp16', 'int8'], required=True)
    args = parser.parse_args()
    convert(args.artifacts, args.profile)
