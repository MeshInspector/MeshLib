public static partial class MR
{
    // fill region located to the left from given edges
    /// Generated from function `MR::fillContourLeft`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> FillContourLeft(MR.Const_MeshTopology topology, MR.Std.Const_Vector_MREdgeId contour)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fillContourLeft_std_vector_MR_EdgeId", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_fillContourLeft_std_vector_MR_EdgeId(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgeId._Underlying *contour);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_fillContourLeft_std_vector_MR_EdgeId(topology._UnderlyingPtr, contour._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::fillContourLeft`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> FillContourLeft(MR.Const_MeshTopology topology, MR.Std.Const_Vector_StdVectorMREdgeId contours)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fillContourLeft_std_vector_std_vector_MR_EdgeId", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_fillContourLeft_std_vector_std_vector_MR_EdgeId(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_StdVectorMREdgeId._Underlying *contours);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_fillContourLeft_std_vector_std_vector_MR_EdgeId(topology._UnderlyingPtr, contours._UnderlyingPtr), is_owning: true));
    }
}
