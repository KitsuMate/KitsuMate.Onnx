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
        /// When true, logs timing for input conversion, inference, and output conversion.
        /// </summary>
        bool VerboseLogging { get; set; }
        
        /// <summary>
        /// Run inference synchronously, copying all inputs/outputs through CPU.
        /// Suitable for single-shot inference (e.g., voice encoder, conditional decoder).
        /// </summary>
        /// <remarks>
        /// <b>Obsolete for multi-step engines.</b> Engines with autoregressive loops
        /// (e.g., language models with KV cache) should cast to <see cref="IOnnxDeviceSession"/>
        /// and use <see cref="IOnnxDeviceSession.RunOnDevice"/> instead. RunOnDevice keeps
        /// intermediate tensors on the GPU, avoiding ~29MB/step of CPU↔GPU round-trips.
        /// </remarks>
        /// <param name="inputs">Dictionary mapping input names to tensors.</param>
        /// <returns>Dictionary mapping output names to result tensors.</returns>
        [Obsolete("For multi-step/autoregressive engines, use IOnnxDeviceSession.RunOnDevice to keep tensors on the GPU between steps.")]
        IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyDictionary<string, OnnxTensor> inputs);
        
        /// <summary>
        /// Run inference with named value containers, copying all inputs/outputs through CPU.
        /// </summary>
        /// <remarks>
        /// <b>Obsolete for multi-step engines.</b> See <see cref="IOnnxDeviceSession.RunOnDevice"/>.
        /// </remarks>
        [Obsolete("For multi-step/autoregressive engines, use IOnnxDeviceSession.RunOnDevice to keep tensors on the GPU between steps.")]
        IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyList<OnnxNamedValue> inputs);
        
        /// <summary>
        /// Run inference asynchronously, copying all inputs/outputs through CPU.
        /// Important for WebGL/JS interop.
        /// </summary>
        /// <remarks>
        /// <b>Obsolete for multi-step engines.</b> See <see cref="IOnnxDeviceSession.RunOnDevice"/>.
        /// </remarks>
        [Obsolete("For multi-step/autoregressive engines, use IOnnxDeviceSession.RunOnDevice to keep tensors on the GPU between steps.")]
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
        /// <para>
        /// <b>Migration guide:</b> If your engine currently calls <c>session.Run()</c> in a
        /// loop and passes outputs back as inputs, refactor as follows:
        /// <list type="number">
        /// <item>Cast session to <see cref="IOnnxDeviceSession"/> (falls back to CPU Run if unavailable).</item>
        /// <item>Pass small/new tensors via <paramref name="cpuInputs"/>, recurrent state via <paramref name="deviceInputs"/>.</item>
        /// <item>List only outputs you need to inspect on CPU in <paramref name="cpuOutputNames"/>;
        ///       everything else stays on device for the next step at zero copy cost.</item>
        /// </list>
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
