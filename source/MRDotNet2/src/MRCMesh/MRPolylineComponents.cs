public static partial class MR
{
    public static partial class PolylineComponents
    {
        /// returns the number of connected components in polyline
        /// Generated from function `MR::PolylineComponents::getNumComponents`.
        public static unsafe ulong GetNumComponents(MR.Const_PolylineTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineComponents_getNumComponents", ExactSpelling = true)]
            extern static ulong __MR_PolylineComponents_getNumComponents(MR.Const_PolylineTopology._Underlying *topology);
            return __MR_PolylineComponents_getNumComponents(topology._UnderlyingPtr);
        }

        /// returns one connected component containing given undirected edge id, 
        /// not effective to call more than once, if several components are needed use getAllComponents
        /// Generated from function `MR::PolylineComponents::getComponent`.
        public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetComponent(MR.Const_PolylineTopology topology, MR.UndirectedEdgeId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineComponents_getComponent", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_PolylineComponents_getComponent(MR.Const_PolylineTopology._Underlying *topology, MR.UndirectedEdgeId id);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_PolylineComponents_getComponent(topology._UnderlyingPtr, id), is_owning: true));
        }

        /// gets all connected components of polyline topology
        /// \note be careful, if mesh is large enough and has many components, the memory overflow will occur
        /// Generated from function `MR::PolylineComponents::getAllComponents`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_MRUndirectedEdgeBitSet> GetAllComponents(MR.Const_PolylineTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineComponents_getAllComponents_1", ExactSpelling = true)]
            extern static MR.Std.Vector_MRUndirectedEdgeBitSet._Underlying *__MR_PolylineComponents_getAllComponents_1(MR.Const_PolylineTopology._Underlying *topology);
            return MR.Misc.Move(new MR.Std.Vector_MRUndirectedEdgeBitSet(__MR_PolylineComponents_getAllComponents_1(topology._UnderlyingPtr), is_owning: true));
        }

        /// gets all connected components of polyline topology
        /// \detail if components  number more than the maxComponentCount, they will be combined into groups of the same size 
        /// \param maxComponentCount should be more then 1
        /// \return pair components bitsets vector and number components in one group if components number more than maxComponentCount
        /// Generated from function `MR::PolylineComponents::getAllComponents`.
        public static unsafe MR.Misc._Moved<MR.Std.Pair_StdVectorMRUndirectedEdgeBitSet_Int> GetAllComponents(MR.Const_PolylineTopology topology, int maxComponentCount)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineComponents_getAllComponents_2", ExactSpelling = true)]
            extern static MR.Std.Pair_StdVectorMRUndirectedEdgeBitSet_Int._Underlying *__MR_PolylineComponents_getAllComponents_2(MR.Const_PolylineTopology._Underlying *topology, int maxComponentCount);
            return MR.Misc.Move(new MR.Std.Pair_StdVectorMRUndirectedEdgeBitSet_Int(__MR_PolylineComponents_getAllComponents_2(topology._UnderlyingPtr, maxComponentCount), is_owning: true));
        }

        /// gets union-find structure for given polyline
        /// Generated from function `MR::PolylineComponents::getUnionFindStructure`.
        public static unsafe MR.Misc._Moved<MR.UnionFind_MRUndirectedEdgeId> GetUnionFindStructure(MR.Const_PolylineTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineComponents_getUnionFindStructure", ExactSpelling = true)]
            extern static MR.UnionFind_MRUndirectedEdgeId._Underlying *__MR_PolylineComponents_getUnionFindStructure(MR.Const_PolylineTopology._Underlying *topology);
            return MR.Misc.Move(new MR.UnionFind_MRUndirectedEdgeId(__MR_PolylineComponents_getUnionFindStructure(topology._UnderlyingPtr), is_owning: true));
        }
    }
}
