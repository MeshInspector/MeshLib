public static partial class MR
{
    /// given path s-v-e, tries to decrease its length by moving away from v
    /// \param outPath intermediate locations between s and e will be added here
    /// \param tmp elements will be temporary allocated here
    /// \param cachePath as far as we need two sides unfold, cache one to reduce allocations
    /// Generated from function `MR::reducePathViaVertex`.
    public static unsafe bool ReducePathViaVertex(MR.Const_Mesh mesh, MR.Const_MeshTriPoint start, MR.VertId v, MR.Const_MeshTriPoint end, MR.Std.Vector_MREdgePoint outPath, MR.Std.Vector_MRVector2f tmp, MR.Std.Vector_MREdgePoint cachePath)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_reducePathViaVertex", ExactSpelling = true)]
        extern static byte __MR_reducePathViaVertex(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *start, MR.VertId v, MR.Const_MeshTriPoint._Underlying *end, MR.Std.Vector_MREdgePoint._Underlying *outPath, MR.Std.Vector_MRVector2f._Underlying *tmp, MR.Std.Vector_MREdgePoint._Underlying *cachePath);
        return __MR_reducePathViaVertex(mesh._UnderlyingPtr, start._UnderlyingPtr, v, end._UnderlyingPtr, outPath._UnderlyingPtr, tmp._UnderlyingPtr, cachePath._UnderlyingPtr) != 0;
    }

    /// converts any input surface path into geodesic path (so reduces its length): start-path-end;
    /// returns actual number of iterations performed
    /// Generated from function `MR::reducePath`.
    /// Parameter `maxIter` defaults to `5`.
    public static unsafe int ReducePath(MR.Const_Mesh mesh, MR.Const_MeshTriPoint start, MR.Std.Vector_MREdgePoint path, MR.Const_MeshTriPoint end, int? maxIter = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_reducePath", ExactSpelling = true)]
        extern static int __MR_reducePath(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *start, MR.Std.Vector_MREdgePoint._Underlying *path, MR.Const_MeshTriPoint._Underlying *end, int *maxIter);
        int __deref_maxIter = maxIter.GetValueOrDefault();
        return __MR_reducePath(mesh._UnderlyingPtr, start._UnderlyingPtr, path._UnderlyingPtr, end._UnderlyingPtr, maxIter.HasValue ? &__deref_maxIter : null);
    }
}
