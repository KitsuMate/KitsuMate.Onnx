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

        /// <summary>
        /// Provider/runtime diagnostics. Creation data remains stable; active-provider and
        /// execution-fallback collections update thread-safely while the session is running.
        /// </summary>
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

        /// <summary>
        /// Allocates a device tensor owned by this session. Its storage stays fixed for its whole
        /// lifetime, so it can serve as a reusable input or as an output written in place by
        /// <see cref="RunBound"/>. The contents start undefined; fill them with
        /// <see cref="IDeviceTensor.CopyFrom"/> before reading.
        /// </summary>
        IDeviceTensor CreateDeviceTensor(string name, OnnxTensorElementType elementType, int[] shape);

        /// <summary>
        /// Runs inference writing each named output into its tensor in place. Keys are graph
        /// input and output names, so one tensor may serve as both an input and an output, for
        /// example a fixed-size KV cache that the graph updates in place.
        /// </summary>
        /// <remarks>
        /// Bind one tensor as both input and output only when the operator that writes it supports
        /// this, such as GroupQueryAttention with a shared KV cache. Other operators would read and
        /// write the same buffer in one dispatch, which WebGPU rejects by losing the device.
        /// When the provider fails and the session can fall back, the session switches provider and
        /// throws <see cref="OnnxProviderFallbackException"/>; <see cref="IDeviceTensor.ToCpu"/> and
        /// <see cref="IDeviceTensor.CopyFrom"/> do the same for tensors this session created.
        /// </remarks>
        /// <param name="graphId">
        /// With <see cref="OnnxSessionOptions.EnableGraphCapture"/>, a non-negative id records this run
        /// once and replays it afterwards; later runs with the same id must bind the same tensors.
        /// -1 runs normally. Providers without graph capture ignore the id.
        /// </param>
        void RunBound(IReadOnlyList<OnnxNamedValue> cpuInputs, IReadOnlyDictionary<string, IDeviceTensor> deviceInputs,
            IReadOnlyDictionary<string, IDeviceTensor> outputs, int graphId = -1);
    }
}
