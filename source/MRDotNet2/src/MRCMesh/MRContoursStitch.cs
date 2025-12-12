public static partial class MR
{
    /// given two contours:
    /// 1) of equal size;
    /// 2) all edges of c0 have no left faces;
    /// 3) all edges of c1 have no right faces;
    /// merge the surface along corresponding edges of two contours, and deletes all vertices and edges from c1
    /// Generated from function `MR::stitchContours`.
    public static unsafe void StitchContours(MR.MeshTopology topology, MR.Std.Const_Vector_MREdgeId c0, MR.Std.Const_Vector_MREdgeId c1)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_stitchContours", ExactSpelling = true)]
        extern static void __MR_stitchContours(MR.MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgeId._Underlying *c0, MR.Std.Const_Vector_MREdgeId._Underlying *c1);
        __MR_stitchContours(topology._UnderlyingPtr, c0._UnderlyingPtr, c1._UnderlyingPtr);
    }

    /// given a closed loop of edges, splits the surface along that loop so that after return:
    /// 1) returned loop has the same size as input, with corresponding edges in same indexed elements of both;
    /// 2) all edges of c0 have no left faces;
    /// 3) all returned edges have no right faces;
    /// Generated from function `MR::cutAlongEdgeLoop`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> CutAlongEdgeLoop(MR.MeshTopology topology, MR.Std.Const_Vector_MREdgeId c0)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_cutAlongEdgeLoop_MR_MeshTopology", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_cutAlongEdgeLoop_MR_MeshTopology(MR.MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgeId._Underlying *c0);
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_cutAlongEdgeLoop_MR_MeshTopology(topology._UnderlyingPtr, c0._UnderlyingPtr), is_owning: true));
    }

    /// given a closed loop of edges, splits the surface along that loop so that after return:
    /// 1) returned loop has the same size as input, with corresponding edges in same indexed elements of both;
    /// 2) all edges of c0 have no left faces;
    /// 3) all returned edges have no right faces;
    /// 4) vertices of the given mesh are updated
    /// Generated from function `MR::cutAlongEdgeLoop`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> CutAlongEdgeLoop(MR.Mesh mesh, MR.Std.Const_Vector_MREdgeId c0)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_cutAlongEdgeLoop_MR_Mesh", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_cutAlongEdgeLoop_MR_Mesh(MR.Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgeId._Underlying *c0);
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_cutAlongEdgeLoop_MR_Mesh(mesh._UnderlyingPtr, c0._UnderlyingPtr), is_owning: true));
    }
}
