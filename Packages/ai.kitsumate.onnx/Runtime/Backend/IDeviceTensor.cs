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
        /// GPU-backed implementations perform a synchronized readback. Bind frequently
        /// read outputs to CPU during RunOnDevice to avoid an extra transfer here.
        /// </summary>
        OnnxTensor ToCpu();
    }

    /// <summary>
    /// Simple CPU-backed device tensor for non-GPU fallback paths.
    /// </summary>
    public sealed class CpuDeviceTensor : IDeviceTensor
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
