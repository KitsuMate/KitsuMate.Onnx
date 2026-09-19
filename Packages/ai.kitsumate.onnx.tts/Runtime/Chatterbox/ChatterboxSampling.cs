using System;
using System.Linq;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    internal static class ChatterboxSampling
    {
        internal static float[] Combine(float[] logits, int vocabulary, int batch, float guidance)
        {
            int sequence = logits.Length / (batch * vocabulary);
            var result = new float[vocabulary];
            int conditional = (sequence - 1) * vocabulary;
            int unconditional = (2 * sequence - 1) * vocabulary;
            for (int i = 0; i < vocabulary; i++)
                result[i] = batch == 1 ? logits[conditional + i]
                    : logits[conditional + i] + guidance * (logits[conditional + i] - logits[unconditional + i]);
            return result;
        }

        internal static int Sample(float[] logits, ChatterboxGenerationConfig config, Random random)
        {
            int[] order = Enumerable.Range(0, logits.Length).ToArray();
            Array.Sort(order, (a, b) => logits[b].CompareTo(logits[a]));
            double maximum = logits[order[0]];
            if (!double.IsFinite(maximum)) throw new InvalidOperationException("Chatterbox produced invalid logits.");
            if (config == null || config.Temperature == 0) return order[0];
            var weights = new double[order.Length];
            double sum = 0;
            int count = 0;
            // Upstream applies min-p first, then calculates top-p from the remaining distribution.
            for (; count < order.Length; count++)
            {
                double weight = Math.Exp((logits[order[count]] - maximum) / config.Temperature);
                if (!double.IsFinite(weight)) throw new InvalidOperationException("Chatterbox produced invalid logits.");
                if (weight < config.MinP) break;
                weights[count] = weight;
                sum += weight;
            }
            double retained = 0;
            int keep = 0;
            do { retained += weights[keep++]; } while (keep < count && retained < config.TopP * sum);
            double draw = random.NextDouble() * retained;
            for (int i = 0; i < keep; i++) if ((draw -= weights[i]) <= 0) return order[i];
            return order[keep - 1];
        }
    }
}
