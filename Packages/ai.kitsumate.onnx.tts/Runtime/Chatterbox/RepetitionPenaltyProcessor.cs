using System;
using System.Collections.Generic;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    /// <summary>
    /// Applies repetition penalty to logits during autoregressive generation.
    /// Penalizes tokens that have already been generated to reduce repetitive output.
    /// Port of HuggingFace's RepetitionPenaltyLogitsProcessor.
    /// </summary>
    public class RepetitionPenaltyProcessor
    {
        private readonly float _penalty;

        /// <param name="penalty">
        /// Penalty factor. Values > 1.0 discourage repetition; values &lt; 1.0 encourage it.
        /// Typical value: 1.2.
        /// </param>
        public RepetitionPenaltyProcessor(float penalty)
        {
            if (penalty <= 0f)
                throw new ArgumentOutOfRangeException(nameof(penalty), "Penalty must be > 0.");
            _penalty = penalty;
        }

        /// <summary>
        /// Applies repetition penalty to the logits in-place.
        /// </summary>
        /// <param name="generatedTokens">All tokens generated so far (flattened, single batch).</param>
        /// <param name="logits">Logits array for the current step (modified in-place).</param>
        public void Apply(IReadOnlyList<int> generatedTokens, float[] logits)
        {
            for (int i = 0; i < generatedTokens.Count; i++)
            {
                int tokenId = generatedTokens[i];
                if (tokenId < 0 || tokenId >= logits.Length) continue;

                float score = logits[tokenId];
                logits[tokenId] = score < 0 ? score * _penalty : score / _penalty;
            }
        }
    }
}
