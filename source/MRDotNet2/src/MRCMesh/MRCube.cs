public static partial class MR
{
    /// creates cube's topology with 8 vertices, 12 triangular faces, 18 undirected edges.
    /// The order of vertices:
    ///   0_v: x=min, y=min, z=min
    ///   1_v: x=min, y=max, z=min
    ///   2_v: x=max, y=max, z=min
    ///   3_v: x=max, y=min, z=min
    ///   4_v: x=min, y=min, z=max
    ///   5_v: x=min, y=max, z=max
    ///   6_v: x=max, y=max, z=max
    ///   7_v: x=max, y=min, z=max
    /// Generated from function `MR::makeCubeTopology`.
    public static unsafe MR.Misc._Moved<MR.MeshTopology> MakeCubeTopology()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeCubeTopology", ExactSpelling = true)]
        extern static MR.MeshTopology._Underlying *__MR_makeCubeTopology();
        return MR.Misc.Move(new MR.MeshTopology(__MR_makeCubeTopology(), is_owning: true));
    }

    /// creates box mesh with given min-corner (base) and given size in every dimension;
    /// with default parameters, creates unit cube mesh with the centroid in (0,0,0)
    /// Generated from function `MR::makeCube`.
    /// Parameter `size` defaults to `Vector3f::diagonal(1.0f)`.
    /// Parameter `base_` defaults to `Vector3f::diagonal(-0.5f)`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeCube(MR.Const_Vector3f? size = null, MR.Const_Vector3f? base_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeCube", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeCube(MR.Const_Vector3f._Underlying *size, MR.Const_Vector3f._Underlying *base_);
        return MR.Misc.Move(new MR.Mesh(__MR_makeCube(size is not null ? size._UnderlyingPtr : null, base_ is not null ? base_._UnderlyingPtr : null), is_owning: true));
    }

    /// creates parallelepiped mesh with given min-corner \p base and given directional vectors \p size
    /// Generated from function `MR::makeParallelepiped`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeParallelepiped(MR.Const_Vector3f? side, MR.Const_Vector3f base_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeParallelepiped", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeParallelepiped(MR.Const_Vector3f._Underlying *side, MR.Const_Vector3f._Underlying *base_);
        return MR.Misc.Move(new MR.Mesh(__MR_makeParallelepiped(side is not null ? side._UnderlyingPtr : null, base_._UnderlyingPtr), is_owning: true));
    }

    /// creates mesh visualizing a box
    /// Generated from function `MR::makeBoxMesh`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeBoxMesh(MR.Const_Box3f box)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeBoxMesh", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeBoxMesh(MR.Const_Box3f._Underlying *box);
        return MR.Misc.Move(new MR.Mesh(__MR_makeBoxMesh(box._UnderlyingPtr), is_owning: true));
    }
}
