namespace KitsuMate.Onnx.Motion
{
    public interface IKimodoConstraintCompiler
    {
        KimodoConditioning Compile(KimodoConstraintSet constraints, int frameCount = KimodoConditioning.DefaultFrameCount);
    }
}
