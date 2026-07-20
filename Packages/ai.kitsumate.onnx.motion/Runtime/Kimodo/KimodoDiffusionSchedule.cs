using System;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    /// <summary>Kimodo's cosine schedule and deterministic eta=0 DDIM update.</summary>
    internal sealed class KimodoDiffusionSchedule
    {
        private const int BaseSteps = 1000;
        private readonly int[] _mappedTimesteps;
        private readonly double[] _alphaCumulative;
        private readonly double[] _alphaCumulativePrevious;

        public int Steps => _mappedTimesteps.Length;

        public KimodoDiffusionSchedule(int steps)
        {
            if (steps < 2 || steps > BaseSteps) throw new ArgumentOutOfRangeException(nameof(steps));

            var baseAlphaCumulative = new double[BaseSteps];
            double cumulative = 1.0;
            for (int i = 0; i < BaseSteps; i++)
            {
                double t1 = (double)i / BaseSteps;
                double t2 = (double)(i + 1) / BaseSteps;
                double beta = Math.Min(1.0 - AlphaBar(t2) / AlphaBar(t1), 0.999);
                cumulative *= 1.0 - beta;
                baseAlphaCumulative[i] = cumulative;
            }

            _mappedTimesteps = new int[steps];
            _alphaCumulative = new double[steps];
            _alphaCumulativePrevious = new double[steps];
            double stride = (double)(BaseSteps - 1) / (steps - 1);
            for (int i = 0; i < steps; i++)
            {
                int mapped = Math.Min(BaseSteps - 1, (int)Math.Round(i * stride, MidpointRounding.ToEven));
                _mappedTimesteps[i] = mapped;
                _alphaCumulative[i] = Math.Max(1e-9, baseAlphaCumulative[mapped]);
                _alphaCumulativePrevious[i] = i == 0 ? 1.0 : _alphaCumulative[i - 1];
            }
        }

        public int GetModelTimestep(int samplingIndex) => _mappedTimesteps[samplingIndex];

        public void Step(float[] current, float[] predictedClean, int samplingIndex, float[] destination)
        {
            double alpha = _alphaCumulative[samplingIndex];
            double previous = _alphaCumulativePrevious[samplingIndex];
            double sqrtAlpha = Math.Sqrt(alpha);
            double sqrtOneMinusAlpha = Math.Sqrt(1.0 - alpha);
            double sqrtPrevious = Math.Sqrt(previous);
            double sqrtOneMinusPrevious = Math.Sqrt(1.0 - previous);

            for (int i = 0; i < current.Length; i++)
            {
                double epsilon = (current[i] - sqrtAlpha * predictedClean[i]) / sqrtOneMinusAlpha;
                destination[i] = (float)(sqrtPrevious * predictedClean[i] + sqrtOneMinusPrevious * epsilon);
            }
        }

        private static double AlphaBar(double t)
        {
            double value = (t + 0.008) / 1.008 * Math.PI / 2.0;
            double cosine = Math.Cos(value);
            return cosine * cosine;
        }
    }
}
