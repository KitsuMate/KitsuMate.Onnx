using System;
using System.Collections.Generic;
using System.Linq;
using KitsuMate.Onnx.Tts.Chatterbox;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Tests
{
    public sealed class ChatterboxStaticLanguageModelTests
    {
        private const int Capacity = 8;

        [Test]
        public void SetupRecreatesTensorsAfterProviderFallback()
        {
            var model = new FakeLanguageModel { FailingRun = 1 };
            using var languageModel = new ChatterboxStaticLanguageModel(new FakeEmbedding(), model, Capacity);

            Assert.That(model.Generations, Has.Count.EqualTo(2));
            Assert.That(model.Generations[0].All(tensor => tensor.Disposed), Is.True,
                "Tensors of the failed provider must be released.");
            Assert.That(languageModel.Prefill(new long[] { 1, 2 }, new long[] { 0, 1 }, Conditioning(), 0.5f),
                Has.Length.EqualTo(ChatterboxStaticLanguageModel.Batch * ChatterboxConstants.SpeechLogitCount));
        }

        [Test]
        public void PrefillRepeatsOnTheNewProvider()
        {
            var model = new FakeLanguageModel { FailingRun = 2 };
            using var languageModel = new ChatterboxStaticLanguageModel(new FakeEmbedding(), model, Capacity);

            float[] logits = languageModel.Prefill(new long[] { 1, 2, 3 }, new long[] { 0, 1, 2 }, Conditioning(), 0.5f);

            Assert.That(logits[0], Is.EqualTo(3f), "The repeated prefill must see its own length.");
            Assert.That(model.Generations, Has.Count.EqualTo(2));
        }

        [Test]
        public void StepRethrowsAfterFallbackAndTheNextRequestWorks()
        {
            var model = new FakeLanguageModel { FailingRun = 3 };
            using var languageModel = new ChatterboxStaticLanguageModel(new FakeEmbedding(), model, Capacity);
            languageModel.Prefill(new long[] { 1, 2 }, new long[] { 0, 1 }, Conditioning(), 0.5f);

            Assert.Throws<OnnxProviderFallbackException>(() => languageModel.Step(7, 0, 0.5f));

            Assert.That(model.Generations, Has.Count.EqualTo(2));
            languageModel.Prefill(new long[] { 1, 2 }, new long[] { 0, 1 }, Conditioning(), 0.5f);
            Assert.That(languageModel.Step(7, 0, 0.5f)[0], Is.EqualTo(3f));
        }

        private static OnnxTensor Conditioning() =>
            OnnxTensor.FromArray(Array.Empty<float>(), new[] { ChatterboxStaticLanguageModel.Batch, 0, 1024 });

        /// <summary>Returns embeddings shaped like the real token embedding graph.</summary>
        private sealed class FakeEmbedding : IOnnxSession
        {
            public IReadOnlyList<string> InputNames { get; } = Array.Empty<string>();
            public IReadOnlyList<string> OutputNames { get; } = new[] { "inputs_embeds" };
            public OnnxSessionDiagnostics Diagnostics => null;
            public bool VerboseLogging { get; set; }

            public IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyDictionary<string, OnnxTensor> inputs)
            {
                int tokens = inputs["input_ids"].Shape[1] + inputs["conditioning"].Shape[1];
                int[] shape = { ChatterboxStaticLanguageModel.Batch, tokens, 1024 };
                return new Dictionary<string, OnnxTensor>
                    { ["inputs_embeds"] = OnnxTensor.FromArray(new float[shape.Aggregate(1, (a, b) => a * b)], shape) };
            }

            public IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyList<OnnxNamedValue> inputs) =>
                Run(inputs.ToDictionary(input => input.Name, input => input.Value));

            public Awaitable<IReadOnlyDictionary<string, OnnxTensor>> RunAsync(IReadOnlyDictionary<string, OnnxTensor> inputs) =>
                throw new NotSupportedException();

            public void Dispose() { }
        }

        /// <summary>
        /// A one-layer language model whose chosen bound run fails like a provider that fell back.
        /// Its logits hold the bound sequence length, so tests can see which run produced them.
        /// </summary>
        private sealed class FakeLanguageModel : IOnnxDeviceSession
        {
            private int runs;

            public int FailingRun { get; set; }
            public List<List<FakeTensor>> Generations { get; } = new() { new List<FakeTensor>() };
            public IReadOnlyList<string> InputNames { get; } = new[]
            {
                "inputs_embeds", "seqlens_k", "total_sequence_length", "past_key_values.0.key", "past_key_values.0.value"
            };
            public IReadOnlyList<string> OutputNames { get; } = new[] { "logits", "present.0.key", "present.0.value" };
            public OnnxSessionDiagnostics Diagnostics => null;
            public bool VerboseLogging { get; set; }

            public IDeviceTensor CreateDeviceTensor(string name, OnnxTensorElementType elementType, int[] shape)
            {
                int count = shape.Aggregate(1, (a, b) => a * b);
                Array data = elementType == OnnxTensorElementType.Int32 ? new int[count] : new float[count];
                var tensor = new FakeTensor(name, elementType, shape, data);
                Generations[^1].Add(tensor);
                return tensor;
            }

            public void RunBound(IReadOnlyList<OnnxNamedValue> cpuInputs, IReadOnlyDictionary<string, IDeviceTensor> deviceInputs,
                IReadOnlyDictionary<string, IDeviceTensor> outputs, int graphId = -1)
            {
                if (++runs == FailingRun)
                {
                    Generations.Add(new List<FakeTensor>());
                    throw new OnnxProviderFallbackException(OnnxExecutionProvider.WebGpu, OnnxExecutionProvider.Cpu,
                        new InvalidOperationException("Simulated provider failure."));
                }
                var tensors = deviceInputs.Values.Concat(outputs.Values).Cast<FakeTensor>();
                Assert.That(tensors.All(tensor => !tensor.Disposed), Is.True, "A bound tensor was already released.");
                int length = ((int[])((FakeTensor)deviceInputs["total_sequence_length"]).Data)[0];
                float[] logits = (float[])((FakeTensor)outputs["logits"]).Data;
                for (int i = 0; i < logits.Length; i++) logits[i] = length;
            }

            public IReadOnlyList<IDeviceTensor> RunOnDevice(IReadOnlyList<OnnxNamedValue> cpuInputs,
                IReadOnlyList<IDeviceTensor> deviceInputs, IReadOnlyCollection<string> cpuOutputNames) =>
                throw new NotSupportedException();

            public IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyDictionary<string, OnnxTensor> inputs) =>
                throw new NotSupportedException();

            public IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyList<OnnxNamedValue> inputs) =>
                throw new NotSupportedException();

            public Awaitable<IReadOnlyDictionary<string, OnnxTensor>> RunAsync(IReadOnlyDictionary<string, OnnxTensor> inputs) =>
                throw new NotSupportedException();

            public void Dispose() { }
        }

        /// <summary>A host tensor that, like a GPU tensor, copies on every transfer.</summary>
        private sealed class FakeTensor : IDeviceTensor
        {
            private readonly OnnxTensorElementType elementType;
            private readonly int[] shape;

            public FakeTensor(string name, OnnxTensorElementType elementType, int[] shape, Array data)
            {
                Name = name;
                this.elementType = elementType;
                this.shape = shape;
                Data = data;
            }

            public string Name { get; set; }
            public Array Data { get; }
            public bool Disposed { get; private set; }

            public OnnxTensor ToCpu() => new(shape, elementType, (Array)Data.Clone(), Name);

            public void CopyFrom(OnnxTensor source)
            {
                Assert.That(Disposed, Is.False);
                Array.Copy(source.Data, Data, Data.Length);
            }

            public void Dispose() => Disposed = true;
        }
    }
}
