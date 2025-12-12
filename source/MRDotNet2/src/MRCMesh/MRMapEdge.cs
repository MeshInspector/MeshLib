public static partial class MR
{
    /// given input edge (src), converts its id using given map
    /// Generated from function `MR::mapEdge`.
    public static unsafe MR.EdgeId MapEdge(MR.Const_WholeEdgeMap map, MR.EdgeId src)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mapEdge_const_MR_WholeEdgeMap_ref_MR_EdgeId", ExactSpelling = true)]
        extern static MR.EdgeId __MR_mapEdge_const_MR_WholeEdgeMap_ref_MR_EdgeId(MR.Const_WholeEdgeMap._Underlying *map, MR.EdgeId src);
        return __MR_mapEdge_const_MR_WholeEdgeMap_ref_MR_EdgeId(map._UnderlyingPtr, src);
    }

    /// given input edge (src), converts its id using given map
    /// Generated from function `MR::mapEdge`.
    public static unsafe MR.UndirectedEdgeId MapEdge(MR.Const_WholeEdgeMap map, MR.UndirectedEdgeId src)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mapEdge_const_MR_WholeEdgeMap_ref_MR_UndirectedEdgeId", ExactSpelling = true)]
        extern static MR.UndirectedEdgeId __MR_mapEdge_const_MR_WholeEdgeMap_ref_MR_UndirectedEdgeId(MR.Const_WholeEdgeMap._Underlying *map, MR.UndirectedEdgeId src);
        return __MR_mapEdge_const_MR_WholeEdgeMap_ref_MR_UndirectedEdgeId(map._UnderlyingPtr, src);
    }

    /// given input edge (src), converts its id using given map
    /// Generated from function `MR::mapEdge`.
    public static unsafe MR.EdgeId MapEdge(MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MREdgeId map, MR.EdgeId src)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mapEdge_const_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_EdgeId", ExactSpelling = true)]
        extern static MR.EdgeId __MR_mapEdge_const_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_EdgeId(MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *map, MR.EdgeId src);
        return __MR_mapEdge_const_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_EdgeId(map._UnderlyingPtr, src);
    }

    /// given input edge (src), converts its id using given map
    /// Generated from function `MR::mapEdge`.
    public static unsafe MR.UndirectedEdgeId MapEdge(MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MREdgeId map, MR.UndirectedEdgeId src)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mapEdge_const_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_UndirectedEdgeId", ExactSpelling = true)]
        extern static MR.UndirectedEdgeId __MR_mapEdge_const_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_UndirectedEdgeId(MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *map, MR.UndirectedEdgeId src);
        return __MR_mapEdge_const_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_UndirectedEdgeId(map._UnderlyingPtr, src);
    }

    /// given input edge (src), converts its id using given map
    /// Generated from function `MR::mapEdge`.
    public static unsafe MR.EdgeId MapEdge(MR.Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId m, MR.EdgeId src)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mapEdge_const_MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_EdgeId", ExactSpelling = true)]
        extern static MR.EdgeId __MR_mapEdge_const_MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_EdgeId(MR.Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *m, MR.EdgeId src);
        return __MR_mapEdge_const_MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_EdgeId(m._UnderlyingPtr, src);
    }

    /// given input edge (src), converts its id using given map
    /// Generated from function `MR::mapEdge`.
    public static unsafe MR.UndirectedEdgeId MapEdge(MR.Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId m, MR.UndirectedEdgeId src)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mapEdge_const_MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_UndirectedEdgeId", ExactSpelling = true)]
        extern static MR.UndirectedEdgeId __MR_mapEdge_const_MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_UndirectedEdgeId(MR.Const_MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *m, MR.UndirectedEdgeId src);
        return __MR_mapEdge_const_MR_MapOrHashMap_MR_UndirectedEdgeId_MR_EdgeId_ref_MR_UndirectedEdgeId(m._UnderlyingPtr, src);
    }

    /// given input edge (src), converts its id using given map
    /// Generated from function `MR::mapEdge`.
    public static unsafe MR.UndirectedEdgeId MapEdge(MR.Const_UndirectedEdgeBMap map, MR.UndirectedEdgeId src)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mapEdge_const_MR_UndirectedEdgeBMap_ref", ExactSpelling = true)]
        extern static MR.UndirectedEdgeId __MR_mapEdge_const_MR_UndirectedEdgeBMap_ref(MR.Const_UndirectedEdgeBMap._Underlying *map, MR.UndirectedEdgeId src);
        return __MR_mapEdge_const_MR_UndirectedEdgeBMap_ref(map._UnderlyingPtr, src);
    }

    /// given input bit-set (src), converts each id corresponding to set bit using given map, and sets its bit in the resulting bit set
    /// Generated from function `MR::mapEdges`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> MapEdges(MR.Const_WholeEdgeMap map, MR.Const_UndirectedEdgeBitSet src)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mapEdges_MR_WholeEdgeMap", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_mapEdges_MR_WholeEdgeMap(MR.Const_WholeEdgeMap._Underlying *map, MR.Const_UndirectedEdgeBitSet._Underlying *src);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_mapEdges_MR_WholeEdgeMap(map._UnderlyingPtr, src._UnderlyingPtr), is_owning: true));
    }

    /// given input bit-set (src), converts each id corresponding to set bit using given map, and sets its bit in the resulting bit set
    /// Generated from function `MR::mapEdges`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> MapEdges(MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MREdgeId map, MR.Const_UndirectedEdgeBitSet src)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mapEdges_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_EdgeId", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_mapEdges_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_EdgeId(MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *map, MR.Const_UndirectedEdgeBitSet._Underlying *src);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_mapEdges_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_EdgeId(map._UnderlyingPtr, src._UnderlyingPtr), is_owning: true));
    }

    /// given input bit-set (src), converts each id corresponding to set bit using given map, and sets its bit in the resulting bit set
    /// Generated from function `MR::mapEdges`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> MapEdges(MR.Const_UndirectedEdgeBMap map, MR.Const_UndirectedEdgeBitSet src)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mapEdges_MR_UndirectedEdgeBMap", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_mapEdges_MR_UndirectedEdgeBMap(MR.Const_UndirectedEdgeBMap._Underlying *map, MR.Const_UndirectedEdgeBitSet._Underlying *src);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_mapEdges_MR_UndirectedEdgeBMap(map._UnderlyingPtr, src._UnderlyingPtr), is_owning: true));
    }
}
