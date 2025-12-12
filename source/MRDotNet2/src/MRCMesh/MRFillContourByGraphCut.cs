public static partial class MR
{
    /**
    * \brief Fills region located to the left from given contour, by minimizing the sum of metric over the boundary
    * If the computations are terminated by \p progress, then returns the best approximation found by the moment of termination
    *
    */
    /// Generated from function `MR::fillContourLeftByGraphCut`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> FillContourLeftByGraphCut(MR.Const_MeshTopology topology, MR.Std.Const_Vector_MREdgeId contour, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fillContourLeftByGraphCut_std_vector_MR_EdgeId", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_fillContourLeftByGraphCut_std_vector_MR_EdgeId(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgeId._Underlying *contour, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_fillContourLeftByGraphCut_std_vector_MR_EdgeId(topology._UnderlyingPtr, contour._UnderlyingPtr, metric._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null), is_owning: true));
    }

    /**
    * \brief Fills region located to the left from given contours, by minimizing the sum of metric over the boundary
    * If the computations are terminated by \p progress, then returns the best approximation found by the moment of termination
    *
    */
    /// Generated from function `MR::fillContourLeftByGraphCut`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> FillContourLeftByGraphCut(MR.Const_MeshTopology topology, MR.Std.Const_Vector_StdVectorMREdgeId contours, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fillContourLeftByGraphCut_std_vector_std_vector_MR_EdgeId", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_fillContourLeftByGraphCut_std_vector_std_vector_MR_EdgeId(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_StdVectorMREdgeId._Underlying *contours, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_fillContourLeftByGraphCut_std_vector_std_vector_MR_EdgeId(topology._UnderlyingPtr, contours._UnderlyingPtr, metric._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null), is_owning: true));
    }

    /**
    * \brief Finds segment that divide mesh on source and sink (source included, sink excluded), by minimizing the sum of metric over the boundary
    * If the computations are terminated by \p progress, then returns the best approximation found by the moment of termination
    *
    */
    /// Generated from function `MR::segmentByGraphCut`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> SegmentByGraphCut(MR.Const_MeshTopology topology, MR.Const_FaceBitSet source, MR.Const_FaceBitSet sink, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_segmentByGraphCut", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_segmentByGraphCut(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *source, MR.Const_FaceBitSet._Underlying *sink, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_segmentByGraphCut(topology._UnderlyingPtr, source._UnderlyingPtr, sink._UnderlyingPtr, metric._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null), is_owning: true));
    }
}
