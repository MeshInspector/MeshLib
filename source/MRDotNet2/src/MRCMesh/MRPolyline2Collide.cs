public static partial class MR
{
    /**
    * \brief finds all pairs of colliding edges from two 2d polylines
    * \param rigidB2A rigid transformation from B-polyline space to A polyline space, nullptr considered as identity transformation
    * \param firstIntersectionOnly if true then the function returns at most one pair of intersecting edges and returns faster
    */
    /// Generated from function `MR::findCollidingEdgePairs`.
    /// Parameter `firstIntersectionOnly` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgePointPair> FindCollidingEdgePairs(MR.Const_Polyline2 a, MR.Const_Polyline2 b, MR.Const_AffineXf2f? rigidB2A = null, bool? firstIntersectionOnly = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCollidingEdgePairs", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgePointPair._Underlying *__MR_findCollidingEdgePairs(MR.Const_Polyline2._Underlying *a, MR.Const_Polyline2._Underlying *b, MR.Const_AffineXf2f._Underlying *rigidB2A, byte *firstIntersectionOnly);
        byte __deref_firstIntersectionOnly = firstIntersectionOnly.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Std.Vector_MREdgePointPair(__MR_findCollidingEdgePairs(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, firstIntersectionOnly.HasValue ? &__deref_firstIntersectionOnly : null), is_owning: true));
    }

    /**
    * \brief finds all pairs of colliding edges from two 2d polylines
    * \param rigidB2A rigid transformation from B-polyline space to A polyline space, nullptr considered as identity transformation
    * \param firstIntersectionOnly if true then the function returns at most one pair of intersecting edges and returns faster
    */
    /// Generated from function `MR::findCollidingEdges`.
    /// Parameter `firstIntersectionOnly` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRUndirectedEdgeUndirectedEdge> FindCollidingEdges(MR.Const_Polyline2 a, MR.Const_Polyline2 b, MR.Const_AffineXf2f? rigidB2A = null, bool? firstIntersectionOnly = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCollidingEdges", ExactSpelling = true)]
        extern static MR.Std.Vector_MRUndirectedEdgeUndirectedEdge._Underlying *__MR_findCollidingEdges(MR.Const_Polyline2._Underlying *a, MR.Const_Polyline2._Underlying *b, MR.Const_AffineXf2f._Underlying *rigidB2A, byte *firstIntersectionOnly);
        byte __deref_firstIntersectionOnly = firstIntersectionOnly.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Std.Vector_MRUndirectedEdgeUndirectedEdge(__MR_findCollidingEdges(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, firstIntersectionOnly.HasValue ? &__deref_firstIntersectionOnly : null), is_owning: true));
    }

    /**
    * \brief finds bitset per polyline with colliding edges
    * \param rigidB2A rigid transformation from B-polyline space to A polyline space, nullptr considered as identity transformation
    */
    /// Generated from function `MR::findCollidingEdgesBitsets`.
    public static unsafe MR.Misc._Moved<MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet> FindCollidingEdgesBitsets(MR.Const_Polyline2 a, MR.Const_Polyline2 b, MR.Const_AffineXf2f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCollidingEdgesBitsets", ExactSpelling = true)]
        extern static MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet._Underlying *__MR_findCollidingEdgesBitsets(MR.Const_Polyline2._Underlying *a, MR.Const_Polyline2._Underlying *b, MR.Const_AffineXf2f._Underlying *rigidB2A);
        return MR.Misc.Move(new MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(__MR_findCollidingEdgesBitsets(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null), is_owning: true));
    }

    /// finds all pairs of colliding edges from 2d polyline
    /// Generated from function `MR::findSelfCollidingEdgePairs`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgePointPair> FindSelfCollidingEdgePairs(MR.Const_Polyline2 polyline)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSelfCollidingEdgePairs", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgePointPair._Underlying *__MR_findSelfCollidingEdgePairs(MR.Const_Polyline2._Underlying *polyline);
        return MR.Misc.Move(new MR.Std.Vector_MREdgePointPair(__MR_findSelfCollidingEdgePairs(polyline._UnderlyingPtr), is_owning: true));
    }

    /// finds all pairs of colliding edges from 2d polyline
    /// Generated from function `MR::findSelfCollidingEdges`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRUndirectedEdgeUndirectedEdge> FindSelfCollidingEdges(MR.Const_Polyline2 polyline)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSelfCollidingEdges", ExactSpelling = true)]
        extern static MR.Std.Vector_MRUndirectedEdgeUndirectedEdge._Underlying *__MR_findSelfCollidingEdges(MR.Const_Polyline2._Underlying *polyline);
        return MR.Misc.Move(new MR.Std.Vector_MRUndirectedEdgeUndirectedEdge(__MR_findSelfCollidingEdges(polyline._UnderlyingPtr), is_owning: true));
    }

    /// finds the union of all self-intersecting edges
    /// Generated from function `MR::findSelfCollidingEdgesBS`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> FindSelfCollidingEdgesBS(MR.Const_Polyline2 polyline)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSelfCollidingEdgesBS", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_findSelfCollidingEdgesBS(MR.Const_Polyline2._Underlying *polyline);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_findSelfCollidingEdgesBS(polyline._UnderlyingPtr), is_owning: true));
    }

    /**
    * \brief checks that arbitrary 2d polyline A is inside of closed 2d polyline B
    * \param rigidB2A rigid transformation from B-polyline space to A polyline space, nullptr considered as identity transformation
    */
    /// Generated from function `MR::isInside`.
    public static unsafe bool IsInside(MR.Const_Polyline2 a, MR.Const_Polyline2 b, MR.Const_AffineXf2f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isInside_MR_Polyline2", ExactSpelling = true)]
        extern static byte __MR_isInside_MR_Polyline2(MR.Const_Polyline2._Underlying *a, MR.Const_Polyline2._Underlying *b, MR.Const_AffineXf2f._Underlying *rigidB2A);
        return __MR_isInside_MR_Polyline2(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null) != 0;
    }
}
