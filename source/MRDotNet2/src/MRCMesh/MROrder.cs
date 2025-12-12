public static partial class MR
{
    /// computes optimal order of faces: old face id -> new face id,
    /// the order is similar as in AABB tree, but faster to compute
    /// Generated from function `MR::getOptimalFaceOrdering`.
    public static unsafe MR.Misc._Moved<MR.FaceBMap> GetOptimalFaceOrdering(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getOptimalFaceOrdering", ExactSpelling = true)]
        extern static MR.FaceBMap._Underlying *__MR_getOptimalFaceOrdering(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FaceBMap(__MR_getOptimalFaceOrdering(mesh._UnderlyingPtr), is_owning: true));
    }

    /// compute the order of vertices given the order of faces:
    /// vertices near first faces also appear first;
    /// \param faceMap old face id -> new face id
    /// Generated from function `MR::getVertexOrdering`.
    public static unsafe MR.Misc._Moved<MR.VertBMap> GetVertexOrdering(MR.Const_FaceBMap faceMap, MR.Const_MeshTopology topology)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getVertexOrdering_MR_FaceBMap", ExactSpelling = true)]
        extern static MR.VertBMap._Underlying *__MR_getVertexOrdering_MR_FaceBMap(MR.Const_FaceBMap._Underlying *faceMap, MR.Const_MeshTopology._Underlying *topology);
        return MR.Misc.Move(new MR.VertBMap(__MR_getVertexOrdering_MR_FaceBMap(faceMap._UnderlyingPtr, topology._UnderlyingPtr), is_owning: true));
    }

    /// compute the order of edges given the order of faces:
    /// edges near first faces also appear first;
    /// \param faceMap old face id -> new face id
    /// Generated from function `MR::getEdgeOrdering`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBMap> GetEdgeOrdering(MR.Const_FaceBMap faceMap, MR.Const_MeshTopology topology)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getEdgeOrdering", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBMap._Underlying *__MR_getEdgeOrdering(MR.Const_FaceBMap._Underlying *faceMap, MR.Const_MeshTopology._Underlying *topology);
        return MR.Misc.Move(new MR.UndirectedEdgeBMap(__MR_getEdgeOrdering(faceMap._UnderlyingPtr, topology._UnderlyingPtr), is_owning: true));
    }
}
