public static partial class MR
{
    /// adds to the region all faces within given number of hops (stars) from the initial region boundary
    /// Generated from function `MR::expand`.
    /// Parameter `hops` defaults to `1`.
    public static unsafe void Expand(MR.Const_MeshTopology topology, MR.FaceBitSet region, int? hops = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expand_MR_FaceBitSet", ExactSpelling = true)]
        extern static void __MR_expand_MR_FaceBitSet(MR.Const_MeshTopology._Underlying *topology, MR.FaceBitSet._Underlying *region, int *hops);
        int __deref_hops = hops.GetValueOrDefault();
        __MR_expand_MR_FaceBitSet(topology._UnderlyingPtr, region._UnderlyingPtr, hops.HasValue ? &__deref_hops : null);
    }

    /// returns the region of all faces within given number of hops (stars) from the initial face
    /// Generated from function `MR::expand`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> Expand(MR.Const_MeshTopology topology, MR.FaceId f, int hops)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expand_MR_FaceId", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_expand_MR_FaceId(MR.Const_MeshTopology._Underlying *topology, MR.FaceId f, int hops);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_expand_MR_FaceId(topology._UnderlyingPtr, f, hops), is_owning: true));
    }

    // adds to the region all vertices within given number of hops (stars) from the initial region boundary
    /// Generated from function `MR::expand`.
    /// Parameter `hops` defaults to `1`.
    public static unsafe void Expand(MR.Const_MeshTopology topology, MR.VertBitSet region, int? hops = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expand_MR_VertBitSet", ExactSpelling = true)]
        extern static void __MR_expand_MR_VertBitSet(MR.Const_MeshTopology._Underlying *topology, MR.VertBitSet._Underlying *region, int *hops);
        int __deref_hops = hops.GetValueOrDefault();
        __MR_expand_MR_VertBitSet(topology._UnderlyingPtr, region._UnderlyingPtr, hops.HasValue ? &__deref_hops : null);
    }

    /// returns the region of all vertices within given number of hops (stars) from the initial vertex
    /// Generated from function `MR::expand`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> Expand(MR.Const_MeshTopology topology, MR.VertId v, int hops)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expand_MR_VertId", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_expand_MR_VertId(MR.Const_MeshTopology._Underlying *topology, MR.VertId v, int hops);
        return MR.Misc.Move(new MR.VertBitSet(__MR_expand_MR_VertId(topology._UnderlyingPtr, v, hops), is_owning: true));
    }

    /// removes from the region all faces within given number of hops (stars) from the initial region boundary
    /// Generated from function `MR::shrink`.
    /// Parameter `hops` defaults to `1`.
    public static unsafe void Shrink(MR.Const_MeshTopology topology, MR.FaceBitSet region, int? hops = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_shrink_MR_FaceBitSet", ExactSpelling = true)]
        extern static void __MR_shrink_MR_FaceBitSet(MR.Const_MeshTopology._Underlying *topology, MR.FaceBitSet._Underlying *region, int *hops);
        int __deref_hops = hops.GetValueOrDefault();
        __MR_shrink_MR_FaceBitSet(topology._UnderlyingPtr, region._UnderlyingPtr, hops.HasValue ? &__deref_hops : null);
    }

    /// removes from the region all vertices within given number of hops (stars) from the initial region boundary
    /// Generated from function `MR::shrink`.
    /// Parameter `hops` defaults to `1`.
    public static unsafe void Shrink(MR.Const_MeshTopology topology, MR.VertBitSet region, int? hops = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_shrink_MR_VertBitSet", ExactSpelling = true)]
        extern static void __MR_shrink_MR_VertBitSet(MR.Const_MeshTopology._Underlying *topology, MR.VertBitSet._Underlying *region, int *hops);
        int __deref_hops = hops.GetValueOrDefault();
        __MR_shrink_MR_VertBitSet(topology._UnderlyingPtr, region._UnderlyingPtr, hops.HasValue ? &__deref_hops : null);
    }

    /// returns given region with all faces sharing an edge with a region face;
    /// \param stopEdges - neighborhood via this edges will be ignored
    /// Generated from function `MR::expandFaces`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> ExpandFaces(MR.Const_MeshTopology topology, MR.Const_FaceBitSet region, MR.Const_UndirectedEdgeBitSet? stopEdges = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expandFaces", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_expandFaces(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region, MR.Const_UndirectedEdgeBitSet._Underlying *stopEdges);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_expandFaces(topology._UnderlyingPtr, region._UnderlyingPtr, stopEdges is not null ? stopEdges._UnderlyingPtr : null), is_owning: true));
    }

    /// returns given region without all faces sharing an edge with not-region face;
    /// \param stopEdges - neighborhood via this edges will be ignored
    /// Generated from function `MR::shrinkFaces`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> ShrinkFaces(MR.Const_MeshTopology topology, MR.Const_FaceBitSet region, MR.Const_UndirectedEdgeBitSet? stopEdges = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_shrinkFaces", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_shrinkFaces(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region, MR.Const_UndirectedEdgeBitSet._Underlying *stopEdges);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_shrinkFaces(topology._UnderlyingPtr, region._UnderlyingPtr, stopEdges is not null ? stopEdges._UnderlyingPtr : null), is_owning: true));
    }

    /// returns faces from given region that have at least one neighbor face with shared edge not from the region
    /// Generated from function `MR::getBoundaryFaces`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetBoundaryFaces(MR.Const_MeshTopology topology, MR.Const_FaceBitSet region)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getBoundaryFaces", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_getBoundaryFaces(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_getBoundaryFaces(topology._UnderlyingPtr, region._UnderlyingPtr), is_owning: true));
    }
}
