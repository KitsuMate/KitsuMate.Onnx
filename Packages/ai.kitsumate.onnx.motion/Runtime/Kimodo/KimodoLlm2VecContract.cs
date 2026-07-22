namespace KitsuMate.Onnx.Motion.Kimodo
{
    internal static class KimodoLlm2VecContract
    {
        public const int SequenceLength = 64;
        public const int EmbeddingDimension = 4096;
        public const long PadTokenId = 128001;
        public const long BeginOfTextTokenId = 128000;
        public const long StartHeaderTokenId = 128006;
        public const long UserTokenId = 882;
        public const long EndHeaderTokenId = 128007;
        public const long DoubleNewlineTokenId = 271;
        public const long EndOfTurnTokenId = 128009;

        public const string InputIds = "input_ids";
        public const string AttentionMask = "attention_mask";
        public const string PoolingMask = "pooling_mask";
        public const string Embedding = "embedding";

        public const string InstructionPrefix = "<|start_header_id|>user<|end_header_id|>\n\n";
        public const string EndOfTurn = "<|eot_id|>";
        public const string ModelId = "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised";
    }
}
