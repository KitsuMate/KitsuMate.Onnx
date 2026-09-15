"""Replace NeuCodec's iSTFT ConvTranspose with an equivalent overlap Conv.

The native WebGPU ConvTranspose kernel drops contributions for the codec's
480-sample stride. This transformation keeps the weights and waveform semantics
while using stride-one Conv, Transpose and Reshape, including dynamic lengths.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def rewrite_istft(model):
    nodes = [n for n in model.graph.node if n.op_type == 'ConvTranspose'
             and len(n.input) == 2 and n.input[1].endswith('istft.inverse_basis')]
    if len(nodes) != 1:
        raise ValueError('Expected one NeuCodec iSTFT ConvTranspose with inverse_basis weights.')
    node = nodes[0]
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    stride = attributes.get('strides', [1])
    if (len(stride) != 1 or stride[0] <= 0 or attributes.get('group', 1) != 1
            or attributes.get('pads', [0, 0]) != [0, 0]
            or attributes.get('dilations', [1]) != [1]
            or attributes.get('output_padding', [0]) != [0]
            or attributes.get('auto_pad', b'NOTSET') != b'NOTSET'
            or 'output_shape' in attributes):
        raise ValueError('Unsupported iSTFT convolution attributes.')
    weight = next(w for w in model.graph.initializer if w.name == node.input[1])
    basis = numpy_helper.to_array(weight)
    hop = stride[0]
    if (basis.ndim != 3 or basis.shape[1] != 1 or basis.shape[2] % hop
            or basis.dtype != np.float32):
        raise ValueError('Expected FP32 inverse_basis [channels, 1, multiple of hop].')
    overlaps = basis.shape[2] // hop
    # y[t*hop+r] = sum_q,c x[c,t-q] * basis[c,0,q*hop+r].
    # Conv indexes its kernel forwards, so reverse the overlap axis.
    packed = basis[:, 0, :].reshape(basis.shape[0], overlaps, hop)
    packed = packed.transpose(2, 0, 1)[:, :, ::-1].copy()
    prefix = node.name + '_overlap'
    model.graph.initializer.extend([
        numpy_helper.from_array(packed, prefix + '_weight'),
        numpy_helper.from_array(np.array([0, 1, -1], np.int64), prefix + '_shape')])
    replacement = [
        helper.make_node('Conv', [node.input[0], prefix + '_weight'], [prefix + '_hops'],
                         name=prefix + '_conv', pads=[overlaps - 1, overlaps - 1],
                         strides=[1], dilations=[1], group=1),
        helper.make_node('Transpose', [prefix + '_hops'], [prefix + '_samples'],
                         name=prefix + '_transpose', perm=[0, 2, 1]),
        helper.make_node('Reshape', [prefix + '_samples', prefix + '_shape'], list(node.output),
                         name=prefix + '_reshape')]
    graph_nodes = list(model.graph.node)
    index = graph_nodes.index(node)
    graph_nodes[index:index + 1] = replacement
    del model.graph.node[:]
    model.graph.node.extend(graph_nodes)
    if not any(weight.name in n.input for n in graph_nodes):
        model.graph.initializer.remove(weight)
    model.doc_string += '\nKitsuMate: replaced iSTFT ConvTranspose with equivalent overlap Conv for WebGPU correctness.'
    onnx.checker.check_model(model)
    return model


def replace_unbind_sequences(model):
    """Use tensor Gather for unbind indexing, avoiding CPU-only sequence operators."""
    splits = {n.output[0]: n for n in model.graph.node if n.op_type == 'SplitToSequence'}
    for name, split in splits.items():
        attributes = {a.name: helper.get_attribute_value(a) for a in split.attribute}
        consumers = [n for n in model.graph.node if name in n.input]
        if (len(split.input) != 1 or attributes.get('keepdims', 1) != 0
                or any(n.op_type != 'SequenceAt' or n.input[0] != name for n in consumers)
                or any(o.name == name for o in model.graph.output)):
            raise ValueError('Expected unbind with only SequenceAt consumers.')
        for node in consumers:
            replacement = helper.make_node('Gather', [split.input[0], node.input[1]],
                                           list(node.output), name=node.name,
                                           axis=attributes.get('axis', 0))
            node.CopyFrom(replacement)
    nodes = [n for n in model.graph.node if n.op_type != 'SplitToSequence']
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    values = [v for v in model.graph.value_info if v.name not in splits]
    del model.graph.value_info[:]
    model.graph.value_info.extend(values)
    model.doc_string += '\nKitsuMate: replaced unbind sequence indexing with tensor Gather to avoid GPU/CPU transfers.'
    onnx.checker.check_model(model)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('Keep the original codec in a separate source file for parity checks.')
    model = replace_unbind_sequences(rewrite_istft(onnx.load(args.source)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(args.output)
