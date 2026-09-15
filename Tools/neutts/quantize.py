"""Create the CPU weight-only INT4 profile from the validated FP32 backbone."""
import argparse
import logging
from pathlib import Path
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    if not (args.artifacts / "export-validation.json").is_file():
        raise RuntimeError("Run FP32 export validation first.")
    logging.getLogger().setLevel(logging.WARNING)
    model = onnx.load(args.artifacts / "onnx/backbone_fp32.onnx")
    model = onnx.version_converter.convert_version(model, 21)
    quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=128, is_symmetric=True,
        accuracy_level=4, op_types_to_quantize=("MatMul", "Gather"),
        quant_axes=(("MatMul", 0), ("Gather", 1)))
    quantizer.process()
    quantizer.model.save_model_to_file(str(args.artifacts / "onnx/backbone_int4.onnx"),
        use_external_data_format=True)
    print("INT4 graph written; audio validation is still required.")
