using System;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    internal static class ChatterboxSampling
    {
        // Scratch buffers for one generation step. Sampling runs on the engine's inference thread
        // and is called once per generated token, so reusing them keeps the loop allocation free.
        [ThreadStatic] private static double[] _order;
        [ThreadStatic] private static int[] _candidates;

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
            int best = 0;
            for (int i = 1; i < logits.Length; i++)
                if (logits[i] > logits[best]) best = i;
            double maximum = logits[best];
            if (!double.IsFinite(maximum)) throw new InvalidOperationException("Chatterbox produced invalid logits.");
            if (config == null || config.Temperature == 0) return best;

            if (_order == null || _order.Length < logits.Length)
            {
                _order = new double[logits.Length];
                _candidates = new int[logits.Length];
            }

            // Min-p keeps only the tokens within a fraction of the most likely one, which is a small
            // set in practice. Weighing every token but ordering only the survivors keeps the step
            // cost linear instead of sorting the whole vocabulary for every generated token.
            int count = 0;
            double sum = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                double weight = Math.Exp((logits[i] - maximum) / config.Temperature);
                if (!double.IsFinite(weight)) throw new InvalidOperationException("Chatterbox produced invalid logits.");
                if (weight < config.MinP) continue;
                _order[count] = -weight;
                _candidates[count] = i;
                sum += weight;
                count++;
            }
            Array.Sort(_order, _candidates, 0, count);

            // Upstream applies min-p first, then calculates top-p from the remaining distribution.
            double retained = 0;
            int keep = 0;
            do { retained -= _order[keep++]; } while (keep < count && retained < config.TopP * sum);
            double draw = random.NextDouble() * retained;
            for (int i = 0; i < keep; i++) if ((draw += _order[i]) <= 0) return _candidates[i];
            return _candidates[keep - 1];
        }
    }
}
