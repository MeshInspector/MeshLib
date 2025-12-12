public static partial class MR
{
    /// Combines unordered input intersections (and flips orientation of intersected edges from mesh B) into ordered oriented contours with the properties:
    /// 1. Each contour is
    ///    a. either closed (then its first and last elements are equal),
    ///    b. or open (then its first and last intersected edges are boundary edges).
    /// 2. Next intersection in a contour is located to the left of the current intersected edge:
    ///    a. if the current and next intersected triangles are the same, then next intersected edge is either next( curr.edge ) or prev( curr.edge.sym() ).sym(),
    ///    b. otherwise next intersected triangle is left( curr.edge ) and next intersected edge is one of the edges having the current intersected triangle to the right.
    /// 3. Orientation of intersected edges in each pair of (intersected edge, intersected triangle):
    ///    a. the intersected edge of mesh A is directed from negative half-space of the intersected triangle from mesh B to its positive half-space,
    ///    b. the intersected edge of mesh B is directed from positive half-space of the intersected triangle from mesh A to its negative half-space.
    /// 4. Orientation of contours:
    ///    a. left  of contours on mesh A is inside of mesh B (consequence of 3a),
    ///    b. right of contours on mesh B is inside of mesh A (consequence of 3b).
    /// Generated from function `MR::orderIntersectionContours`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVarEdgeTri> OrderIntersectionContours(MR.Const_MeshTopology topologyA, MR.Const_MeshTopology topologyB, MR.Std.Const_Vector_MRVarEdgeTri intersections)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orderIntersectionContours", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVarEdgeTri._Underlying *__MR_orderIntersectionContours(MR.Const_MeshTopology._Underlying *topologyA, MR.Const_MeshTopology._Underlying *topologyB, MR.Std.Const_Vector_MRVarEdgeTri._Underlying *intersections);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVarEdgeTri(__MR_orderIntersectionContours(topologyA._UnderlyingPtr, topologyB._UnderlyingPtr, intersections._UnderlyingPtr), is_owning: true));
    }

    /// Combines unordered input self-intersections (and flips orientation of some intersected edges) into ordered oriented contours with the properties:
    /// 1. Each contour is
    ///    a. either closed (then its first and last elements are equal),
    ///    b. or open if terminal intersection is on mesh boundary or if self-intersection terminates in a vertex.
    /// 2. Next intersection in a contour is located to the left of the current intersected edge:
    ///    a. if the current and next intersected triangles are the same, then next intersected edge is either next( curr.edge ) or prev( curr.edge.sym() ).sym(),
    ///    b. otherwise next intersected triangle is left( curr.edge ) and next intersected edge is one of the edges having the current intersected triangle to the right.
    /// 3. Orientation of intersected edges in each pair of (intersected edge, intersected triangle):
    ///    a. isEdgeATriB() = true:  the intersected edge is directed from negative half-space of the intersected triangle to its positive half-space,
    ///    b. isEdgeATriB() = false: the intersected edge is directed from positive half-space of the intersected triangle to its negative half-space.
    /// 4. Contours [2*i] and [2*i+1]
    ///    a. have equal lengths and pass via the same intersections but in opposite order,
    ///    b. each intersection is present in two contours with different values of isEdgeATriB() flag, and opposite directions of the intersected edge.
    /// 5. Orientation of contours:
    ///    a. first element of even (0,2,...) contours has isEdgeATriB() = true, left of even contours goes inside (consequence of 3a),
    ///    b. first element of odd (1,3,...) contours has isEdgeATriB() = false, right of odd contours goes inside (consequence of 3b).
    /// Generated from function `MR::orderSelfIntersectionContours`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVarEdgeTri> OrderSelfIntersectionContours(MR.Const_MeshTopology topology, MR.Std.Const_Vector_MREdgeTri intersections)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orderSelfIntersectionContours", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVarEdgeTri._Underlying *__MR_orderSelfIntersectionContours(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgeTri._Underlying *intersections);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVarEdgeTri(__MR_orderSelfIntersectionContours(topology._UnderlyingPtr, intersections._UnderlyingPtr), is_owning: true));
    }

    /// returns true if contour is closed
    /// Generated from function `MR::isClosed`.
    public static unsafe bool IsClosed(MR.Std.Const_Vector_MRVarEdgeTri contour)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isClosed_std_vector_MR_VarEdgeTri", ExactSpelling = true)]
        extern static byte __MR_isClosed_std_vector_MR_VarEdgeTri(MR.Std.Const_Vector_MRVarEdgeTri._Underlying *contour);
        return __MR_isClosed_std_vector_MR_VarEdgeTri(contour._UnderlyingPtr) != 0;
    }

    /// Detects contours that fully lay inside one triangle
    /// if `ignoreOpen` then do not mark non-closed contours as lone, even if they really are
    /// returns they indices in contours
    /// Generated from function `MR::detectLoneContours`.
    /// Parameter `ignoreOpen` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_Int> DetectLoneContours(MR.Std.Const_Vector_StdVectorMRVarEdgeTri contours, bool? ignoreOpen = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_detectLoneContours", ExactSpelling = true)]
        extern static MR.Std.Vector_Int._Underlying *__MR_detectLoneContours(MR.Std.Const_Vector_StdVectorMRVarEdgeTri._Underlying *contours, byte *ignoreOpen);
        byte __deref_ignoreOpen = ignoreOpen.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Std.Vector_Int(__MR_detectLoneContours(contours._UnderlyingPtr, ignoreOpen.HasValue ? &__deref_ignoreOpen : null), is_owning: true));
    }

    /// Removes contours with zero area (do not remove if contour is handle on topology)
    /// edgesTopology - topology on which contours are represented with edges
    /// faceContours - lone contours represented by faces (all intersections are in same mesh A face)
    /// edgeContours - lone contours represented by edges (all intersections are in mesh B edges, edgesTopology: meshB.topology)
    /// Generated from function `MR::removeLoneDegeneratedContours`.
    public static unsafe void RemoveLoneDegeneratedContours(MR.Const_MeshTopology edgesTopology, MR.Std.Vector_MROneMeshContour faceContours, MR.Std.Vector_MROneMeshContour edgeContours)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_removeLoneDegeneratedContours", ExactSpelling = true)]
        extern static void __MR_removeLoneDegeneratedContours(MR.Const_MeshTopology._Underlying *edgesTopology, MR.Std.Vector_MROneMeshContour._Underlying *faceContours, MR.Std.Vector_MROneMeshContour._Underlying *edgeContours);
        __MR_removeLoneDegeneratedContours(edgesTopology._UnderlyingPtr, faceContours._UnderlyingPtr, edgeContours._UnderlyingPtr);
    }

    /// Removes contours that fully lay inside one triangle from the contours
    /// if `ignoreOpen` then do not consider non-closed contours as lone, even if they really are
    /// Generated from function `MR::removeLoneContours`.
    /// Parameter `ignoreOpen` defaults to `false`.
    public static unsafe void RemoveLoneContours(MR.Std.Vector_StdVectorMRVarEdgeTri contours, bool? ignoreOpen = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_removeLoneContours", ExactSpelling = true)]
        extern static void __MR_removeLoneContours(MR.Std.Vector_StdVectorMRVarEdgeTri._Underlying *contours, byte *ignoreOpen);
        byte __deref_ignoreOpen = ignoreOpen.GetValueOrDefault() ? (byte)1 : (byte)0;
        __MR_removeLoneContours(contours._UnderlyingPtr, ignoreOpen.HasValue ? &__deref_ignoreOpen : null);
    }
}
