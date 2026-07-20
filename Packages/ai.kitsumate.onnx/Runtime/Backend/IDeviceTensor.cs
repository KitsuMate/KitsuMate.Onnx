using System;

namespace KitsuMate.Onnx
{
    /// <summary>
    /// Represents a tensor held on the inference device (e.g., GPU or CPU).
    /// Avoids copying data between CPU and device memory when passing outputs
    /// back as inputs in autoregressive loops.
    /// </summary>
    public interface IDeviceTensor : IDisposable
    {
        /// <summary>Name of this tensor (for binding as named input).</summary>
        string Name { get; set; }

        /// <summary>
        /// Copy tensor data from device to CPU.
        /// Works for CPU-resident tensors. For GPU-resident tensors, must be
        /// bound as a CPU output during RunOnDevice.
        /// </summary>
        OnnxTensor ToCpu();
    }

    /// <summary>
    /// Simple CPU-backed device tensor for non-GPU fallback paths.
    /// </summary>
    internal class CpuDeviceTensor : IDeviceTensor
    {
        private OnnxTensor _tensor;

        public string Name { get; set; }

        public CpuDeviceTensor(OnnxTensor tensor)
        {
            _tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
            Name = tensor.Name;
        }

        public OnnxTensor ToCpu() => _tensor;

        public void Dispose()
        {
            _tensor?.Dispose();
            _tensor = null;
        }
    }
}
