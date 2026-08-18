using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx
{
    /// <summary>
    /// Represents an ONNX inference session that can run models.
    /// Backend-agnostic interface allowing different implementations
    /// (ONNX Runtime, Unity Inference Engine, Transformers.js, etc.)
    /// </summary>
    public interface IOnnxSession : IDisposable
    {
        /// <summary>Names of all input tensors expected by the model.</summary>
        IReadOnlyList<string> InputNames { get; }
        
        /// <summary>Names of all output tensors produced by the model.</summary>
        IReadOnlyList<string> OutputNames { get; }

        /// <summary>Immutable provider and runtime information captured when the session was created.</summary>
        OnnxSessionDiagnostics Diagnostics { get; }
        
        /// <summary>
        /// When true, logs timing for input conversion, inference, and output conversion.
        /// </summary>
        bool VerboseLogging { get; set; }
        
        /// <summary>
        /// Run inference synchronously, copying all inputs/outputs through CPU.
        /// Suitable for single-shot inference (e.g., voice encoder, conditional decoder).
        /// </summary>
        /// <param name="inputs">Dictionary mapping input names to tensors.</param>
        /// <returns>Dictionary mapping output names to result tensors.</returns>
        IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyDictionary<string, OnnxTensor> inputs);
        
        /// <summary>
        /// Run inference with named value containers, copying all inputs/outputs through CPU.
        /// </summary>
        IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyList<OnnxNamedValue> inputs);
        
        /// <summary>
        /// Run inference asynchronously, copying all inputs/outputs through CPU.
        /// Important for WebGL/JS interop.
        /// </summary>
        Awaitable<IReadOnlyDictionary<string, OnnxTensor>> RunAsync(IReadOnlyDictionary<string, OnnxTensor> inputs);
    }

    /// <summary>
    /// Extended session interface for device-resident tensor operations.
    /// Enables keeping tensors (e.g., KV cache) on the inference device between
    /// autoregressive steps, avoiding redundant CPU↔device copies.
    /// </summary>
    public interface IOnnxDeviceSession : IOnnxSession
    {
        /// <summary>
        /// Run inference keeping outputs as device-resident tensors.
        /// <b>Preferred path for multi-step / autoregressive engines</b> (language models,
        /// diffusion loops, any pipeline that feeds outputs back as inputs).
        /// </summary>
        /// <remarks>
        /// <para>
        /// Unlike <see cref="IOnnxSession.Run"/>, this method uses IO Binding to keep
        /// intermediate tensors (KV cache, hidden states) on the inference device between
        /// steps, eliminating expensive CPU↔GPU memory copies on every iteration.
        /// </para>
        /// </remarks>
        /// <param name="cpuInputs">Inputs provided as CPU tensors (small or new each step).</param>
        /// <param name="deviceInputs">Inputs from previous RunOnDevice calls (stay on device — zero copy).</param>
        /// <param name="cpuOutputNames">
        /// Output names to place on CPU (accessible via <see cref="IDeviceTensor.ToCpu"/>).
        /// All other outputs stay on device for passthrough to next call.
        /// If null, all outputs are placed on CPU.
        /// </param>
        /// <returns>All outputs as device tensors, ordered by session output names.</returns>
        IReadOnlyList<IDeviceTensor> RunOnDevice(
            IReadOnlyList<OnnxNamedValue> cpuInputs,
            IReadOnlyList<IDeviceTensor> deviceInputs,
            IReadOnlyCollection<string> cpuOutputNames);
    }
}
