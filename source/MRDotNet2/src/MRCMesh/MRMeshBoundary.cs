public static partial class MR
{
    /// adds triangles along the boundary to straighten it;
    /// \details new triangle is added only if 
    ///  1) aspect ratio of the new triangle is at most maxTriAspectRatio,
    ///  2) dot product of its normal with neighbor triangles is at least minNeiNormalsDot.
    /// Generated from function `MR::straightenBoundary`.
    public static unsafe void StraightenBoundary(MR.Mesh mesh, MR.EdgeId bd, float minNeiNormalsDot, float maxTriAspectRatio, MR.FaceBitSet? newFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_straightenBoundary", ExactSpelling = true)]
        extern static void __MR_straightenBoundary(MR.Mesh._Underlying *mesh, MR.EdgeId bd, float minNeiNormalsDot, float maxTriAspectRatio, MR.FaceBitSet._Underlying *newFaces);
        __MR_straightenBoundary(mesh._UnderlyingPtr, bd, minNeiNormalsDot, maxTriAspectRatio, newFaces is not null ? newFaces._UnderlyingPtr : null);
    }
}
