public static partial class MR
{
    /// Generated from function `MR::rayBoxIntersect<float>`.
    public static unsafe bool RayBoxIntersect(MR.Const_Box2f box, MR.Const_Line2f line, float t0, float t1)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayBoxIntersect_4_MR_Box2f", ExactSpelling = true)]
        extern static byte __MR_rayBoxIntersect_4_MR_Box2f(MR.Const_Box2f._Underlying *box, MR.Const_Line2f._Underlying *line, float t0, float t1);
        return __MR_rayBoxIntersect_4_MR_Box2f(box._UnderlyingPtr, line._UnderlyingPtr, t0, t1) != 0;
    }
}
