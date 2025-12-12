public static partial class MR
{
    /// returns a map where each valid vertex is mapped to the smallest valid vertex Id located within given distance (including itself), and this smallest vertex is mapped to itself,
    /// each vertex not from valid set is mapped to itself
    /// Generated from function `MR::findSmallestCloseVertices`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertMap> FindSmallestCloseVertices(MR.Const_Mesh mesh, float closeDist, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSmallestCloseVertices_3_MR_Mesh", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertMap._Underlying *__MR_findSmallestCloseVertices_3_MR_Mesh(MR.Const_Mesh._Underlying *mesh, float closeDist, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Optional_MRVertMap(__MR_findSmallestCloseVertices_3_MR_Mesh(mesh._UnderlyingPtr, closeDist, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }

    /// returns a map where each valid vertex is mapped to the smallest valid vertex Id located within given distance (including itself), and this smallest vertex is mapped to itself,
    /// each vertex not from valid set is mapped to itself
    /// Generated from function `MR::findSmallestCloseVertices`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertMap> FindSmallestCloseVertices(MR.Const_PointCloud cloud, float closeDist, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSmallestCloseVertices_3_MR_PointCloud", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertMap._Underlying *__MR_findSmallestCloseVertices_3_MR_PointCloud(MR.Const_PointCloud._Underlying *cloud, float closeDist, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Optional_MRVertMap(__MR_findSmallestCloseVertices_3_MR_PointCloud(cloud._UnderlyingPtr, closeDist, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }

    /// returns a map where each valid vertex is mapped to the smallest valid vertex Id located within given distance (including itself), and this smallest vertex is mapped to itself,
    /// each vertex not from valid set is mapped to itself; the search tree is constructe inside
    /// Generated from function `MR::findSmallestCloseVertices`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertMap> FindSmallestCloseVertices(MR.Const_VertCoords points, float closeDist, MR.Const_VertBitSet? valid = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSmallestCloseVertices_4", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertMap._Underlying *__MR_findSmallestCloseVertices_4(MR.Const_VertCoords._Underlying *points, float closeDist, MR.Const_VertBitSet._Underlying *valid, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Optional_MRVertMap(__MR_findSmallestCloseVertices_4(points._UnderlyingPtr, closeDist, valid is not null ? valid._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }

    /// returns a map where each valid vertex is mapped to the smallest valid vertex Id located within given distance (including itself), and this smallest vertex is mapped to itself,
    /// each vertex not from valid set is mapped to itself; given tree is used as is
    /// Generated from function `MR::findSmallestCloseVerticesUsingTree`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertMap> FindSmallestCloseVerticesUsingTree(MR.Const_VertCoords points, float closeDist, MR.Const_AABBTreePoints tree, MR.Const_VertBitSet? valid, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSmallestCloseVerticesUsingTree", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertMap._Underlying *__MR_findSmallestCloseVerticesUsingTree(MR.Const_VertCoords._Underlying *points, float closeDist, MR.Const_AABBTreePoints._Underlying *tree, MR.Const_VertBitSet._Underlying *valid, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Optional_MRVertMap(__MR_findSmallestCloseVerticesUsingTree(points._UnderlyingPtr, closeDist, tree._UnderlyingPtr, valid is not null ? valid._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }

    /// finds all close vertices, where for each vertex there is another one located within given distance
    /// Generated from function `MR::findCloseVertices`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertBitSet> FindCloseVertices(MR.Const_Mesh mesh, float closeDist, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCloseVertices_3_MR_Mesh", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_findCloseVertices_3_MR_Mesh(MR.Const_Mesh._Underlying *mesh, float closeDist, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Optional_MRVertBitSet(__MR_findCloseVertices_3_MR_Mesh(mesh._UnderlyingPtr, closeDist, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }

    /// finds all close vertices, where for each vertex there is another one located within given distance
    /// Generated from function `MR::findCloseVertices`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertBitSet> FindCloseVertices(MR.Const_PointCloud cloud, float closeDist, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCloseVertices_3_MR_PointCloud", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_findCloseVertices_3_MR_PointCloud(MR.Const_PointCloud._Underlying *cloud, float closeDist, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Optional_MRVertBitSet(__MR_findCloseVertices_3_MR_PointCloud(cloud._UnderlyingPtr, closeDist, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }

    /// finds all close vertices, where for each vertex there is another one located within given distance
    /// Generated from function `MR::findCloseVertices`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertBitSet> FindCloseVertices(MR.Const_VertCoords points, float closeDist, MR.Const_VertBitSet? valid = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCloseVertices_4", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_findCloseVertices_4(MR.Const_VertCoords._Underlying *points, float closeDist, MR.Const_VertBitSet._Underlying *valid, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Optional_MRVertBitSet(__MR_findCloseVertices_4(points._UnderlyingPtr, closeDist, valid is not null ? valid._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }

    /// finds all close vertices, where for each vertex there is another one located within given distance; smallestMap is the result of findSmallestCloseVertices function call
    /// Generated from function `MR::findCloseVertices`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> FindCloseVertices(MR.Const_VertMap smallestMap)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCloseVertices_1", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_findCloseVertices_1(MR.Const_VertMap._Underlying *smallestMap);
        return MR.Misc.Move(new MR.VertBitSet(__MR_findCloseVertices_1(smallestMap._UnderlyingPtr), is_owning: true));
    }

    /// finds pairs of twin edges (each twin edge will be present at least in one of pairs)
    /// Generated from function `MR::findTwinEdgePairs`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdPairMREdgeIdMREdgeId> FindTwinEdgePairs(MR.Const_Mesh mesh, float closeDist)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTwinEdgePairs", ExactSpelling = true)]
        extern static MR.Std.Vector_StdPairMREdgeIdMREdgeId._Underlying *__MR_findTwinEdgePairs(MR.Const_Mesh._Underlying *mesh, float closeDist);
        return MR.Misc.Move(new MR.Std.Vector_StdPairMREdgeIdMREdgeId(__MR_findTwinEdgePairs(mesh._UnderlyingPtr, closeDist), is_owning: true));
    }

    /// finds all directed twin edges
    /// Generated from function `MR::findTwinEdges`.
    public static unsafe MR.Misc._Moved<MR.EdgeBitSet> FindTwinEdges(MR.Const_Mesh mesh, float closeDist)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTwinEdges_2", ExactSpelling = true)]
        extern static MR.EdgeBitSet._Underlying *__MR_findTwinEdges_2(MR.Const_Mesh._Underlying *mesh, float closeDist);
        return MR.Misc.Move(new MR.EdgeBitSet(__MR_findTwinEdges_2(mesh._UnderlyingPtr, closeDist), is_owning: true));
    }

    /// Generated from function `MR::findTwinEdges`.
    public static unsafe MR.Misc._Moved<MR.EdgeBitSet> FindTwinEdges(MR.Std.Const_Vector_StdPairMREdgeIdMREdgeId pairs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTwinEdges_1", ExactSpelling = true)]
        extern static MR.EdgeBitSet._Underlying *__MR_findTwinEdges_1(MR.Std.Const_Vector_StdPairMREdgeIdMREdgeId._Underlying *pairs);
        return MR.Misc.Move(new MR.EdgeBitSet(__MR_findTwinEdges_1(pairs._UnderlyingPtr), is_owning: true));
    }

    /// finds all undirected twin edges
    /// Generated from function `MR::findTwinUndirectedEdges`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> FindTwinUndirectedEdges(MR.Const_Mesh mesh, float closeDist)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTwinUndirectedEdges_2", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_findTwinUndirectedEdges_2(MR.Const_Mesh._Underlying *mesh, float closeDist);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_findTwinUndirectedEdges_2(mesh._UnderlyingPtr, closeDist), is_owning: true));
    }

    /// Generated from function `MR::findTwinUndirectedEdges`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> FindTwinUndirectedEdges(MR.Std.Const_Vector_StdPairMREdgeIdMREdgeId pairs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTwinUndirectedEdges_1", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_findTwinUndirectedEdges_1(MR.Std.Const_Vector_StdPairMREdgeIdMREdgeId._Underlying *pairs);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_findTwinUndirectedEdges_1(pairs._UnderlyingPtr), is_owning: true));
    }

    /// provided that each edge has at most one twin, composes bidirectional mapping between twins
    /// Generated from function `MR::findTwinUndirectedEdgeHashMap`.
    public static unsafe MR.Misc._Moved<MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId> FindTwinUndirectedEdgeHashMap(MR.Const_Mesh mesh, float closeDist)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTwinUndirectedEdgeHashMap_2", ExactSpelling = true)]
        extern static MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_findTwinUndirectedEdgeHashMap_2(MR.Const_Mesh._Underlying *mesh, float closeDist);
        return MR.Misc.Move(new MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(__MR_findTwinUndirectedEdgeHashMap_2(mesh._UnderlyingPtr, closeDist), is_owning: true));
    }

    /// Generated from function `MR::findTwinUndirectedEdgeHashMap`.
    public static unsafe MR.Misc._Moved<MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId> FindTwinUndirectedEdgeHashMap(MR.Std.Const_Vector_StdPairMREdgeIdMREdgeId pairs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTwinUndirectedEdgeHashMap_1", ExactSpelling = true)]
        extern static MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_findTwinUndirectedEdgeHashMap_1(MR.Std.Const_Vector_StdPairMREdgeIdMREdgeId._Underlying *pairs);
        return MR.Misc.Move(new MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId(__MR_findTwinUndirectedEdgeHashMap_1(pairs._UnderlyingPtr), is_owning: true));
    }
}
