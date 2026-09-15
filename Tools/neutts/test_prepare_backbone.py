import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from prepare_backbone import fuse_norms, prepare, prune


def norm_model(exponent=2, axis=-1):
    prefix = '/model/norm/'
    output = lambda name: prefix+name+'_output_0'
    nodes = []
    for name, data in [('Constant', exponent), ('Constant_1', 1e-6), ('Constant_2', 1)]:
        nodes.append(helper.make_node('Constant', [], [output(name)], name=prefix+name,
            value=numpy_helper.from_array(np.array(data, np.float32))))
    for name, op, inputs, attrs in [
        ('Cast', 'Cast', ['x'], dict(to=1)),
        ('Pow', 'Pow', [output('Cast'), output('Constant')], {}),
        ('ReduceMean', 'ReduceMean', [output('Pow')], dict(axes=[axis], keepdims=1)),
        ('Add', 'Add', [output('ReduceMean'), output('Constant_1')], {}),
        ('Sqrt', 'Sqrt', [output('Add')], {}),
        ('Div', 'Div', [output('Constant_2'), output('Sqrt')], {}),
        ('Mul', 'Mul', [output('Cast'), output('Div')], {}),
        ('Cast_1', 'Cast', [output('Mul')], dict(to=1)),
        ('Mul_1', 'Mul', ['scale', output('Cast_1')], {})]:
        nodes.append(helper.make_node(op, inputs, [output(name)], name=prefix+name, **attrs))
    model = helper.make_model(helper.make_graph(nodes, 'norm',
        [helper.make_tensor_value_info('x', 1, [1, 'sequence', 128])],
        [helper.make_tensor_value_info(output('Mul_1'), 1, [1, 'sequence', 128]),
         helper.make_tensor_value_info(output('Cast'), 1, [1, 'sequence', 128])],
        [numpy_helper.from_array(np.linspace(0.5, 1.5, 128).astype(np.float32), 'scale')]),
        opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    return model


class BackbonePreparationTests(unittest.TestCase):
    def test_fused_norm_matches_and_preserves_residual_input(self):
        model = norm_model()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        reference = ort.InferenceSession(model.SerializeToString(), options)
        self.assertEqual(fuse_norms(model), 1)
        prune(model)
        actual = ort.InferenceSession(model.SerializeToString(), options)
        rng = np.random.default_rng(7)
        for sequence in [1, 9, 128]:
            for magnitude in [0, 0.0001, 1, 100]:
                inputs = {'x': rng.normal(size=(1, sequence, 128)).astype(np.float32)*magnitude}
                for a, b in zip(actual.run(None, inputs), reference.run(None, inputs)):
                    np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)

    def test_rejects_different_normalization(self):
        for model in [norm_model(exponent=3), norm_model(axis=1)]:
            with self.assertRaises(ValueError):
                fuse_norms(model)

    def test_rejects_unrelated_backbone(self):
        with self.assertRaises(ValueError):
            prepare(norm_model())


if __name__ == '__main__':
    unittest.main()
