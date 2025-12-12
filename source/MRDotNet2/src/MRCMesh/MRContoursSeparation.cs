public static partial class MR
{
    // Separates mesh into disconnected by contour components (independent components are not returned),
    // faces that are intersected by contour does not belong to any component.
    // Calls callback for each MeshEdgePoint in contour respecting order, 
    // ignoring MeshTriPoints (if projection of input point lay inside face)
    /// Generated from function `MR::separateClosedContour`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRFaceBitSet> SeparateClosedContour(MR.Const_Mesh mesh, MR.Std.Const_Vector_MRVector3f contour, MR.Std.Const_Function_VoidFuncFromConstMREdgePointRef? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_separateClosedContour", ExactSpelling = true)]
        extern static MR.Std.Vector_MRFaceBitSet._Underlying *__MR_separateClosedContour(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MRVector3f._Underlying *contour, MR.Std.Const_Function_VoidFuncFromConstMREdgePointRef._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Vector_MRFaceBitSet(__MR_separateClosedContour(mesh._UnderlyingPtr, contour._UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }
}
