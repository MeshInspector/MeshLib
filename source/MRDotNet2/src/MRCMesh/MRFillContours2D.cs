public static partial class MR
{
    /**
    * @brief fill holes with border in same plane (i.e. after cut by plane)
    * @param mesh - mesh with holes
    * @param holeRepresentativeEdges - each edge here represents a hole borders that should be filled
    * should be not empty
    * edges should have invalid left face (FaceId == -1)
    * @return Expected with has_value()=true if holes filled, otherwise - string error
    */
    /// Generated from function `MR::fillContours2D`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> FillContours2D(MR.Mesh mesh, MR.Std.Const_Vector_MREdgeId holeRepresentativeEdges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fillContours2D", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_fillContours2D(MR.Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgeId._Underlying *holeRepresentativeEdges);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_fillContours2D(mesh._UnderlyingPtr, holeRepresentativeEdges._UnderlyingPtr), is_owning: true));
    }

    /// computes the transformation that maps
    /// O into center mass of contours' points
    /// OXY into best plane containing the points
    /// Generated from function `MR::getXfFromOxyPlane`.
    public static unsafe MR.AffineXf3f GetXfFromOxyPlane(MR.Std.Const_Vector_StdVectorMRVector3f contours)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getXfFromOxyPlane_1", ExactSpelling = true)]
        extern static MR.AffineXf3f __MR_getXfFromOxyPlane_1(MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *contours);
        return __MR_getXfFromOxyPlane_1(contours._UnderlyingPtr);
    }

    /// Generated from function `MR::getXfFromOxyPlane`.
    public static unsafe MR.AffineXf3f GetXfFromOxyPlane(MR.Const_Mesh mesh, MR.Std.Const_Vector_StdVectorMREdgeId paths)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getXfFromOxyPlane_2", ExactSpelling = true)]
        extern static MR.AffineXf3f __MR_getXfFromOxyPlane_2(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_StdVectorMREdgeId._Underlying *paths);
        return __MR_getXfFromOxyPlane_2(mesh._UnderlyingPtr, paths._UnderlyingPtr);
    }

    /// given an ObjectMeshData and the contours of a planar hole in it,
    /// fills the hole using fillContours2D function and updates all per-element attributes;
    /// if some contours were not closed on input, then it closes them by adding a bridge edge in each
    /// Generated from function `MR::fillPlanarHole`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> FillPlanarHole(MR.ObjectMeshData data, MR.Std.Vector_StdVectorMREdgeId holeContours)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fillPlanarHole", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_fillPlanarHole(MR.ObjectMeshData._Underlying *data, MR.Std.Vector_StdVectorMREdgeId._Underlying *holeContours);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_fillPlanarHole(data._UnderlyingPtr, holeContours._UnderlyingPtr), is_owning: true));
    }
}
