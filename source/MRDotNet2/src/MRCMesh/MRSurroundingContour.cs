public static partial class MR
{
    /**
    * \brief Find the best closed edge loop passing through given edges, which minimizes the sum of given edge metric.
    * The algorithm assumes that input edges can be projected on the plane orthogonal to given direction,
    * then the center point of all input edges is found, and each segment of the searched loop is within infinite pie sector
    * with this center and the borders passing via two sorted input edges.
    *
    * 
    * \param includeEdges contain all edges in arbitrary order that must be present in the returned loop, probably with reversed direction (should have at least 2 elements)
    * \param edgeMetric returned loop will minimize the sum of this metric
    * \param dir direction approximately orthogonal to the loop, the resulting loop will be oriented clockwise if look from the direction's tip
    */
    /// Generated from function `MR::surroundingContour`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMREdgeId_StdString> SurroundingContour(MR.Const_Mesh mesh, MR.Std._ByValue_Vector_MREdgeId includeEdges, MR.Std.Const_Function_FloatFuncFromMREdgeId edgeMetric, MR.Const_Vector3f dir)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_surroundingContour_std_vector_MR_EdgeId", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMREdgeId_StdString._Underlying *__MR_surroundingContour_std_vector_MR_EdgeId(MR.Const_Mesh._Underlying *mesh, MR.Misc._PassBy includeEdges_pass_by, MR.Std.Vector_MREdgeId._Underlying *includeEdges, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *edgeMetric, MR.Const_Vector3f._Underlying *dir);
        return MR.Misc.Move(new MR.Expected_StdVectorMREdgeId_StdString(__MR_surroundingContour_std_vector_MR_EdgeId(mesh._UnderlyingPtr, includeEdges.PassByMode, includeEdges.Value is not null ? includeEdges.Value._UnderlyingPtr : null, edgeMetric._UnderlyingPtr, dir._UnderlyingPtr), is_owning: true));
    }

    /**
    * \brief Find the best closed edge loop passing through given vertices, which minimizes the sum of given edge metric.
    * The algorithm assumes that input vertices can be projected on the plane orthogonal to given direction,
    * then the center point of all input vertices is found, and each segment of the searched loop is within infinite pie sector
    * with this center and the borders passing via two sorted input vertices.
    *
    * 
    * \param keyVertices contain all vertices in arbitrary order that returned loop must pass (should have at least 2 elements)
    * \param edgeMetric returned loop will minimize the sum of this metric
    * \param dir direction approximately orthogonal to the loop, the resulting loop will be oriented clockwise if look from the direction's tip
    */
    /// Generated from function `MR::surroundingContour`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMREdgeId_StdString> SurroundingContour(MR.Const_Mesh mesh, MR.Std._ByValue_Vector_MRVertId keyVertices, MR.Std.Const_Function_FloatFuncFromMREdgeId edgeMetric, MR.Const_Vector3f dir)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_surroundingContour_std_vector_MR_VertId", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMREdgeId_StdString._Underlying *__MR_surroundingContour_std_vector_MR_VertId(MR.Const_Mesh._Underlying *mesh, MR.Misc._PassBy keyVertices_pass_by, MR.Std.Vector_MRVertId._Underlying *keyVertices, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *edgeMetric, MR.Const_Vector3f._Underlying *dir);
        return MR.Misc.Move(new MR.Expected_StdVectorMREdgeId_StdString(__MR_surroundingContour_std_vector_MR_VertId(mesh._UnderlyingPtr, keyVertices.PassByMode, keyVertices.Value is not null ? keyVertices.Value._UnderlyingPtr : null, edgeMetric._UnderlyingPtr, dir._UnderlyingPtr), is_owning: true));
    }
}
