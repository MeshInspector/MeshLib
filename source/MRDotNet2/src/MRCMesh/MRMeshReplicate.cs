public static partial class MR
{
    /// adjusts z-coordinates of (m) vertices to make adjusted (m) similar to (target)
    /// Generated from function `MR::replicateZ`.
    public static unsafe void ReplicateZ(MR.Mesh m, MR.Const_Mesh target)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_replicateZ", ExactSpelling = true)]
        extern static void __MR_replicateZ(MR.Mesh._Underlying *m, MR.Const_Mesh._Underlying *target);
        __MR_replicateZ(m._UnderlyingPtr, target._UnderlyingPtr);
    }
}
