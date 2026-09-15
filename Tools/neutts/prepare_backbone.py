"""Prepare the NeuTTS export for fewer WebGPU dispatches, preserving FP32 arithmetic.

Uses ONNX Runtime's SimplifiedLayerNormalization and GroupQueryAttention extensions.
The model retains the existing logits and KV-cache interface. No weights are quantized.
Attention is causal, as in NeuTtsEngineRuntime; attention_mask supplies sequence lengths.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def prune(model):
    needed = {o.name for o in model.graph.output}
    nodes = []
    for node in reversed(model.graph.node):
        if any(name in needed for name in node.output):
            nodes.append(node)
            needed.update(node.input)
    weights = [w for w in model.graph.initializer if w.name in needed]
    del model.graph.node[:]
    model.graph.node.extend(reversed(nodes))
    del model.graph.initializer[:]
    model.graph.initializer.extend(weights)
    # Exported intermediate annotations can refer to removed values.
    del model.graph.value_info[:]


def fuse_norms(model):
    """Recognize the pinned export's complete RMSNorm formula before replacing it."""
    groups = {n.name.rsplit('/', 1)[0]: [] for n in model.graph.node
              if n.op_type == 'ReduceMean' and 'norm/' in n.name}
    for node in model.graph.node:
        prefix = node.name.rsplit('/', 1)[0]
        if prefix in groups:
            groups[prefix].append(node)
    replacements, removed = {}, set()
    for prefix, nodes in groups.items():
        by = {n.name.rsplit('/', 1)[1]: n for n in nodes}
        def constant(name):
            n = by[name]
            if n.op_type != 'Constant':
                raise ValueError(f'Unexpected RMSNorm constant: {prefix}/{name}')
            return numpy_helper.to_array(next(a.t for a in n.attribute if a.name == 'value'))
        def check(name, op, inputs):
            if by[name].op_type != op or list(by[name].input) != inputs:
                raise ValueError(f'Unexpected RMSNorm formula: {prefix}/{name}')
        output = lambda name: by[name].output[0]
        check('Pow', 'Pow', [output('Cast'), output('Constant')])
        check('ReduceMean', 'ReduceMean', [output('Pow')])
        attrs = {a.name: helper.get_attribute_value(a) for a in by['ReduceMean'].attribute}
        if attrs.get('axes') != [-1] or attrs.get('keepdims', 1) != 1:
            raise ValueError(f'Unsupported RMSNorm axes: {prefix}')
        check('Add', 'Add', [output('ReduceMean'), output('Constant_1')])
        check('Sqrt', 'Sqrt', [output('Add')])
        check('Div', 'Div', [output('Constant_2'), output('Sqrt')])
        check('Mul', 'Mul', [output('Cast'), output('Div')])
        check('Cast_1', 'Cast', [output('Mul')])
        check('Mul_1', 'Mul', [by['Mul_1'].input[0], output('Cast_1')])
        if float(constant('Constant')) != 2 or float(constant('Constant_2')) != 1:
            raise ValueError(f'Unsupported RMSNorm exponent or reciprocal: {prefix}')
        for name in ['Cast', 'Cast_1']:
            if by[name].op_type != 'Cast' or next(a.i for a in by[name].attribute if a.name == 'to') != 1:
                raise ValueError('RMSNorm preparation requires FP32 computation.')
        epsilon = float(constant('Constant_1'))
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('Invalid RMSNorm epsilon.')
        replacements[by['Mul_1'].name] = helper.make_node('SimplifiedLayerNormalization',
            [by['Cast'].input[0], by['Mul_1'].input[0]], list(by['Mul_1'].output),
            name=prefix+'/FusedRmsNorm', epsilon=epsilon, axis=-1, stash_type=1)
        # The input cast is also consumed by the residual branch in this export.
        removed.update(n.name for n in nodes if n.name != by['Cast'].name)
    rewritten = []
    for node in model.graph.node:
        if node.name in replacements:
            rewritten.append(replacements[node.name])
        elif node.name not in removed:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return len(groups)


def simplify_attention(model):
    """Use the export's fixed head dimensions; sequence lengths remain dynamic."""
    nodes = {n.name: n for n in model.graph.node}
    if model.graph.input[0].name != 'input_ids' or model.graph.input[0].type.tensor_type.shape.dim[0].dim_value != 1:
        raise ValueError('Expected the batch-one NeuTTS export.')
    def value(name, values):
        model.graph.initializer.append(numpy_helper.from_array(np.array(values, np.int64), name))
        return name
    for layer in range(28):
        prefix = f'/model/layers.{layer}/self_attn/'
        for suffix, shape in [('Reshape', [0, 0, 12, 128]), ('Reshape_1', [0, 0, 4, 128]),
                              ('Reshape_2', [0, 0, 4, 128])]:
            node = nodes[prefix+suffix]
            if node.op_type != 'Reshape':
                raise ValueError(f'Unexpected attention layout: {node.name}')
            node.input[1] = value(node.name+'_fixed_shape', shape)


def fuse_attention(model):
    """Fuse causal grouped attention, RoPE and cache concatenation together."""
    nodes = {n.name: n for n in model.graph.node}
    frequency_node = next(n for n in model.graph.node if 'onnx::Expand_370' in n.output)
    frequencies = numpy_helper.to_array(next(a.t for a in frequency_node.attribute if a.name == 'value')).reshape(-1)
    if frequencies.shape != (64,) or not np.isfinite(frequencies).all():
        raise ValueError('Unexpected rotary frequencies.')
    angles = np.arange(2048, dtype=np.float32)[:, None] * frequencies[None, :]
    constants = [('gqa_cos', np.cos(angles)), ('gqa_sin', np.sin(angles)),
                 ('gqa_axis', np.array(3, np.int64)), ('gqa_one', np.array([1], np.int32)),
                 ('gqa_unsqueeze', np.array([0], np.int64)),
                 ('gqa_qshape', np.array([0, 0, 1536], np.int64)),
                 ('gqa_kshape', np.array([0, 0, 512], np.int64))]
    for name, data in constants:
        model.graph.initializer.append(numpy_helper.from_array(data, name))
    extra = [helper.make_node('Shape', ['attention_mask'], ['gqa_mask_shape']),
             helper.make_node('Gather', ['gqa_mask_shape', 'gqa_axis'], ['gqa_total64'], axis=0),
             helper.make_node('Cast', ['gqa_total64'], ['gqa_total'], to=6),
             helper.make_node('Unsqueeze', ['gqa_total', 'gqa_unsqueeze'], ['gqa_total1']),
             helper.make_node('Sub', ['gqa_total1', 'gqa_one'], ['gqa_seqlens'])]
    for layer in range(28):
        prefix = f'/model/layers.{layer}/self_attn/'
        extra.extend([
            helper.make_node('Reshape', [nodes[prefix+'q_norm/FusedRmsNorm'].output[0], 'gqa_qshape'], [prefix+'gqa_q']),
            helper.make_node('Reshape', [nodes[prefix+'k_norm/FusedRmsNorm'].output[0], 'gqa_kshape'], [prefix+'gqa_k'])])
        node = nodes[prefix+'Reshape_7']
        node.CopyFrom(helper.make_node('GroupQueryAttention',
            [prefix+'gqa_q', prefix+'gqa_k', nodes[prefix+'v_proj/MatMul'].output[0],
             f'past_key_values.{layer}.key', f'past_key_values.{layer}.value',
             'gqa_seqlens', 'gqa_total', 'gqa_cos', 'gqa_sin'],
            list(node.output)+[f'present.{layer}.key', f'present.{layer}.value'],
            name=prefix+'FusedGQA', domain='com.microsoft', num_heads=12, kv_num_heads=4,
            do_rotary=1, rotary_interleaved=0))
    pending = extra + [n for n in model.graph.node if not
                      (n.op_type == 'Concat' and any(o.startswith('present.') for o in n.output))]
    ready = {w.name for w in model.graph.initializer} | {v.name for v in model.graph.input}
    ordered = []
    while pending:
        batch = [n for n in pending if all(not name or name in ready for name in n.input)]
        if not batch:
            raise ValueError('Unresolved attention graph inputs.')
        for node in batch:
            ordered.append(node)
            ready.update(node.output)
            pending.remove(node)
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    if not any(o.domain == 'com.microsoft' for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid('com.microsoft', 1))


def prepare(model):
    # Check the fixed dimensions before using the pinned export's node layout.
    caches = [v for v in model.graph.input if v.name.startswith('past_key_values.')]
    if len(caches) != 56 or any([d.dim_value for d in v.type.tensor_type.shape.dim][1::2] != [4, 128] for v in caches):
        raise ValueError('Expected NeuTTS with 28 layers, four KV heads and head dimension 128.')
    if fuse_norms(model) != 113:
        raise ValueError('Expected 113 RMSNorm blocks in the NeuTTS export.')
    simplify_attention(model)
    fuse_attention(model)
    prune(model)
    model.doc_string = 'Modified by KitsuMate: fused FP32 RMSNorm, causal GQA and cached RoPE. See LICENSE and ATTRIBUTION.md.'
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('Use a separate output path to preserve the source model.')
    model = prepare(onnx.load(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f'Wrote {args.output}. Run backbone and audio validation before deployment.')

