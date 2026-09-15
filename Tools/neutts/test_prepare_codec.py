import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from prepare_codec import rewrite_istft, replace_unbind_sequences


class CodecRewriteTests(unittest.TestCase):
    def test_matches_original_for_batches_lengths_and_hops(self):
        random = np.random.default_rng(42)
        for hop in [1, 3, 480]:
            basis = random.normal(size=(6, 1, hop * 4)).astype(np.float32)
            graph = helper.make_graph([
                helper.make_node('ConvTranspose', ['x', 'istft.inverse_basis'], ['audio'],
                                 name='istft', strides=[hop])], 'istft',
                [helper.make_tensor_value_info('x', 1, ['batch', 6, 'frames'])],
                [helper.make_tensor_value_info('audio', 1, ['batch', 1, 'samples'])],
                [numpy_helper.from_array(basis, 'istft.inverse_basis')])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
            model.ir_version = 9
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            reference = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
            rewritten = rewrite_istft(model)
            actual = ort.InferenceSession(rewritten.SerializeToString(), options, providers=['CPUExecutionProvider'])
            for batch, length in [(1, 2), (1, 7), (2, 16)]:
                with self.subTest(hop=hop, batch=batch, length=length):
                    inputs = {'x': random.normal(size=(batch, 6, length)).astype(np.float32)}
                    np.testing.assert_allclose(actual.run(None, inputs)[0], reference.run(None, inputs)[0],
                                               atol=1e-5, rtol=1e-5)

    def test_rejects_unrelated_graph(self):
        model = helper.make_model(helper.make_graph([], 'unrelated', [], []))
        with self.assertRaises(ValueError):
            rewrite_istft(model)

    def test_unbind_replacement_matches_sequence_indexing(self):
        nodes = [helper.make_node('SplitToSequence', ['x'], ['parts'], axis=0, keepdims=0)]
        indices = [0, 1, -1]
        nodes.extend(helper.make_node('SequenceAt', ['parts', f'index{i}'], [f'y{i}']) for i in range(3))
        graph = helper.make_graph(nodes, 'unbind',
            [helper.make_tensor_value_info('x', 1, [3, 'batch', 'frames', 4])],
            [helper.make_tensor_value_info(f'y{i}', 1, ['batch', 'frames', 4]) for i in range(3)],
            [numpy_helper.from_array(np.array(value, np.int64), f'index{i}') for i, value in enumerate(indices)])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        model.ir_version = 9
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        reference = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
        converted = replace_unbind_sequences(model)
        actual = ort.InferenceSession(converted.SerializeToString(), options, providers=['CPUExecutionProvider'])
        random = np.random.default_rng(42)
        for batch, frames in [(1, 2), (2, 7)]:
            inputs = {'x': random.normal(size=(3, batch, frames, 4)).astype(np.float32)}
            for a, b in zip(actual.run(None, inputs), reference.run(None, inputs)):
                np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
