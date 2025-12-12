public static partial class MR
{
    /// returns closed loop of region boundary starting from given region boundary edge (region faces on the left, and not-region faces or holes on the right);
    /// if more than two boundary edges connect in one vertex, then the function makes the most abrupt turn to right
    /// Generated from function `MR::trackLeftBoundaryLoop`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> TrackLeftBoundaryLoop(MR.Const_MeshTopology topology, MR.EdgeId e0, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_trackLeftBoundaryLoop_MR_EdgeId", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_trackLeftBoundaryLoop_MR_EdgeId(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId e0, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_trackLeftBoundaryLoop_MR_EdgeId(topology._UnderlyingPtr, e0, region is not null ? region._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::trackLeftBoundaryLoop`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> TrackLeftBoundaryLoop(MR.Const_MeshTopology topology, MR.Const_FaceBitSet region, MR.EdgeId e0)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_trackLeftBoundaryLoop_MR_FaceBitSet", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_trackLeftBoundaryLoop_MR_FaceBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region, MR.EdgeId e0);
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_trackLeftBoundaryLoop_MR_FaceBitSet(topology._UnderlyingPtr, region._UnderlyingPtr, e0), is_owning: true));
    }

    /// returns closed loop of region boundary starting from given region boundary edge (region faces on the right, and not-region faces or holes on the left);
    /// if more than two boundary edges connect in one vertex, then the function makes the most abrupt turn to left
    /// Generated from function `MR::trackRightBoundaryLoop`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> TrackRightBoundaryLoop(MR.Const_MeshTopology topology, MR.EdgeId e0, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_trackRightBoundaryLoop_MR_EdgeId", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_trackRightBoundaryLoop_MR_EdgeId(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId e0, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_trackRightBoundaryLoop_MR_EdgeId(topology._UnderlyingPtr, e0, region is not null ? region._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::trackRightBoundaryLoop`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> TrackRightBoundaryLoop(MR.Const_MeshTopology topology, MR.Const_FaceBitSet region, MR.EdgeId e0)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_trackRightBoundaryLoop_MR_FaceBitSet", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_trackRightBoundaryLoop_MR_FaceBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region, MR.EdgeId e0);
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_trackRightBoundaryLoop_MR_FaceBitSet(topology._UnderlyingPtr, region._UnderlyingPtr, e0), is_owning: true));
    }

    /// returns all region boundary loops;
    /// every loop has region faces on the left, and not-region faces or holes on the right
    /// Generated from function `MR::findLeftBoundary`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgeId> FindLeftBoundary(MR.Const_MeshTopology topology, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findLeftBoundary", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgeId._Underlying *__MR_findLeftBoundary(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgeId(__MR_findLeftBoundary(topology._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
    }

    /// returns all region boundary loops;
    /// every loop has region faces on the right, and not-region faces or holes on the left
    /// Generated from function `MR::findRightBoundary`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgeId> FindRightBoundary(MR.Const_MeshTopology topology, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findRightBoundary", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgeId._Underlying *__MR_findRightBoundary(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgeId(__MR_findRightBoundary(topology._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
    }

    /// deletes all region faces, inner edges and vertices, but keeps boundary edges and vertices of the region (or whole mesh if region is null);
    /// if `keepLoneHoles` is set - keeps boundary even if it has no valid faces on other side
    /// returns edge loops, each having deleted region faces on the left, and not-region faces or holes on the right
    /// Generated from function `MR::delRegionKeepBd`.
    /// Parameter `keepLoneHoles` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgeId> DelRegionKeepBd(MR.Mesh mesh, MR.Const_FaceBitSet? region = null, bool? keepLoneHoles = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_delRegionKeepBd", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgeId._Underlying *__MR_delRegionKeepBd(MR.Mesh._Underlying *mesh, MR.Const_FaceBitSet._Underlying *region, byte *keepLoneHoles);
        byte __deref_keepLoneHoles = keepLoneHoles.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgeId(__MR_delRegionKeepBd(mesh._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, keepLoneHoles.HasValue ? &__deref_keepLoneHoles : null), is_owning: true));
    }

    /// returns all region boundary paths;
    /// every path has region faces on the left, and valid not-region faces on the right
    /// Generated from function `MR::findLeftBoundaryInsideMesh`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgeId> FindLeftBoundaryInsideMesh(MR.Const_MeshTopology topology, MR.Const_FaceBitSet region)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findLeftBoundaryInsideMesh", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgeId._Underlying *__MR_findLeftBoundaryInsideMesh(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgeId(__MR_findLeftBoundaryInsideMesh(topology._UnderlyingPtr, region._UnderlyingPtr), is_owning: true));
    }

    /// returns all region boundary edges, where each edge has a region face on one side, and a valid not-region face on another side
    /// Generated from function `MR::findRegionBoundaryUndirectedEdgesInsideMesh`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> FindRegionBoundaryUndirectedEdgesInsideMesh(MR.Const_MeshTopology topology, MR.Const_FaceBitSet region)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findRegionBoundaryUndirectedEdgesInsideMesh", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_findRegionBoundaryUndirectedEdgesInsideMesh(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_findRegionBoundaryUndirectedEdgesInsideMesh(topology._UnderlyingPtr, region._UnderlyingPtr), is_owning: true));
    }

    /// \returns All out of region faces that have a common edge with at least one region face
    /// Generated from function `MR::findRegionOuterFaces`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> FindRegionOuterFaces(MR.Const_MeshTopology topology, MR.Const_FaceBitSet region)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findRegionOuterFaces", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_findRegionOuterFaces(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_findRegionOuterFaces(topology._UnderlyingPtr, region._UnderlyingPtr), is_owning: true));
    }

    /// composes the set of all vertices incident to given faces
    /// Generated from function `MR::getIncidentVerts`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> GetIncidentVerts(MR.Const_MeshTopology topology, MR.Const_FaceBitSet faces)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getIncidentVerts_2_MR_FaceBitSet", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_getIncidentVerts_2_MR_FaceBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *faces);
        return MR.Misc.Move(new MR.VertBitSet(__MR_getIncidentVerts_2_MR_FaceBitSet(topology._UnderlyingPtr, faces._UnderlyingPtr), is_owning: true));
    }

    /// if faces-parameter is null pointer then simply returns the reference on all valid vertices;
    /// otherwise performs store = getIncidentVerts( topology, *faces ) and returns reference on store
    /// Generated from function `MR::getIncidentVerts`.
    public static unsafe MR.Const_VertBitSet GetIncidentVerts(MR.Const_MeshTopology topology, MR.Const_FaceBitSet? faces, MR.VertBitSet store)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getIncidentVerts_3_const_MR_FaceBitSet_ptr", ExactSpelling = true)]
        extern static MR.Const_VertBitSet._Underlying *__MR_getIncidentVerts_3_const_MR_FaceBitSet_ptr(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *faces, MR.VertBitSet._Underlying *store);
        return new(__MR_getIncidentVerts_3_const_MR_FaceBitSet_ptr(topology._UnderlyingPtr, faces is not null ? faces._UnderlyingPtr : null, store._UnderlyingPtr), is_owning: false);
    }

    /// composes the set of all vertices not on the boundary of a hole and with all their adjacent faces in given set
    /// Generated from function `MR::getInnerVerts`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> GetInnerVerts(MR.Const_MeshTopology topology, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getInnerVerts_const_MR_FaceBitSet_ptr", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_getInnerVerts_const_MR_FaceBitSet_ptr(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.VertBitSet(__MR_getInnerVerts_const_MR_FaceBitSet_ptr(topology._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
    }

    /// composes the set of all boundary vertices for given region (or whole mesh if !region)
    /// Generated from function `MR::getBoundaryVerts`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> GetBoundaryVerts(MR.Const_MeshTopology topology, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getBoundaryVerts", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_getBoundaryVerts(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.VertBitSet(__MR_getBoundaryVerts(topology._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
    }

    /// composes the set of all boundary vertices for given region,
    /// unlike getBoundaryVerts the vertices of mesh boundary having no incident not-region faces are not returned
    /// Generated from function `MR::getRegionBoundaryVerts`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> GetRegionBoundaryVerts(MR.Const_MeshTopology topology, MR.Const_FaceBitSet region)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getRegionBoundaryVerts", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_getRegionBoundaryVerts(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.VertBitSet(__MR_getRegionBoundaryVerts(topology._UnderlyingPtr, region._UnderlyingPtr), is_owning: true));
    }

    /// composes the set of all faces incident to given vertices
    /// Generated from function `MR::getIncidentFaces`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetIncidentFaces(MR.Const_MeshTopology topology, MR.Const_VertBitSet verts)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getIncidentFaces_MR_VertBitSet", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_getIncidentFaces_MR_VertBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertBitSet._Underlying *verts);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_getIncidentFaces_MR_VertBitSet(topology._UnderlyingPtr, verts._UnderlyingPtr), is_owning: true));
    }

    /// composes the set of all faces with all their vertices in given set
    /// Generated from function `MR::getInnerFaces`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetInnerFaces(MR.Const_MeshTopology topology, MR.Const_VertBitSet verts)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getInnerFaces", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_getInnerFaces(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertBitSet._Underlying *verts);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_getInnerFaces(topology._UnderlyingPtr, verts._UnderlyingPtr), is_owning: true));
    }

    /// composes the set of all edges, having a face from given set at the left
    /// Generated from function `MR::getRegionEdges`.
    public static unsafe MR.Misc._Moved<MR.EdgeBitSet> GetRegionEdges(MR.Const_MeshTopology topology, MR.Const_FaceBitSet faces)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getRegionEdges", ExactSpelling = true)]
        extern static MR.EdgeBitSet._Underlying *__MR_getRegionEdges(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *faces);
        return MR.Misc.Move(new MR.EdgeBitSet(__MR_getRegionEdges(topology._UnderlyingPtr, faces._UnderlyingPtr), is_owning: true));
    }

    /// composes the set of all undirected edges, having a face from given set from one of two sides
    /// Generated from function `MR::getIncidentEdges`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetIncidentEdges(MR.Const_MeshTopology topology, MR.Const_FaceBitSet faces)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getIncidentEdges_MR_FaceBitSet", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_getIncidentEdges_MR_FaceBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *faces);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_getIncidentEdges_MR_FaceBitSet(topology._UnderlyingPtr, faces._UnderlyingPtr), is_owning: true));
    }

    /// composes the set of all undirected edges, having at least one common vertex with an edge from given set
    /// Generated from function `MR::getIncidentEdges`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetIncidentEdges(MR.Const_MeshTopology topology, MR.Const_UndirectedEdgeBitSet edges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getIncidentEdges_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_getIncidentEdges_MR_UndirectedEdgeBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_UndirectedEdgeBitSet._Underlying *edges);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_getIncidentEdges_MR_UndirectedEdgeBitSet(topology._UnderlyingPtr, edges._UnderlyingPtr), is_owning: true));
    }

    /// composes the set of all vertices incident to given edges
    /// Generated from function `MR::getIncidentVerts`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> GetIncidentVerts(MR.Const_MeshTopology topology, MR.Const_UndirectedEdgeBitSet edges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getIncidentVerts_2_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_getIncidentVerts_2_MR_UndirectedEdgeBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_UndirectedEdgeBitSet._Underlying *edges);
        return MR.Misc.Move(new MR.VertBitSet(__MR_getIncidentVerts_2_MR_UndirectedEdgeBitSet(topology._UnderlyingPtr, edges._UnderlyingPtr), is_owning: true));
    }

    /// composes the set of all faces incident to given edges
    /// Generated from function `MR::getIncidentFaces`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetIncidentFaces(MR.Const_MeshTopology topology, MR.Const_UndirectedEdgeBitSet edges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getIncidentFaces_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_getIncidentFaces_MR_UndirectedEdgeBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_UndirectedEdgeBitSet._Underlying *edges);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_getIncidentFaces_MR_UndirectedEdgeBitSet(topology._UnderlyingPtr, edges._UnderlyingPtr), is_owning: true));
    }

    /// composes the set of all left and right faces of given edges
    /// Generated from function `MR::getNeighborFaces`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetNeighborFaces(MR.Const_MeshTopology topology, MR.Const_UndirectedEdgeBitSet edges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getNeighborFaces", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_getNeighborFaces(MR.Const_MeshTopology._Underlying *topology, MR.Const_UndirectedEdgeBitSet._Underlying *edges);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_getNeighborFaces(topology._UnderlyingPtr, edges._UnderlyingPtr), is_owning: true));
    }

    /// composes the set of all edges with all their vertices in given set
    /// Generated from function `MR::getInnerEdges`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetInnerEdges(MR.Const_MeshTopology topology, MR.Const_VertBitSet verts)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getInnerEdges_MR_VertBitSet", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_getInnerEdges_MR_VertBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertBitSet._Underlying *verts);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_getInnerEdges_MR_VertBitSet(topology._UnderlyingPtr, verts._UnderlyingPtr), is_owning: true));
    }

    /// composes the set of all edges having both left and right in given region
    /// Generated from function `MR::getInnerEdges`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetInnerEdges(MR.Const_MeshTopology topology, MR.Const_FaceBitSet region)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getInnerEdges_MR_FaceBitSet", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_getInnerEdges_MR_FaceBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_getInnerEdges_MR_FaceBitSet(topology._UnderlyingPtr, region._UnderlyingPtr), is_owning: true));
    }

    /// if edges-parameter is null pointer then simply returns the reference on all valid vertices;
    /// otherwise performs store = getIncidentVerts( topology, *edges ) and returns reference on store
    /// Generated from function `MR::getIncidentVerts`.
    public static unsafe MR.Const_VertBitSet GetIncidentVerts(MR.Const_MeshTopology topology, MR.Const_UndirectedEdgeBitSet? edges, MR.VertBitSet store)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getIncidentVerts_3_const_MR_UndirectedEdgeBitSet_ptr", ExactSpelling = true)]
        extern static MR.Const_VertBitSet._Underlying *__MR_getIncidentVerts_3_const_MR_UndirectedEdgeBitSet_ptr(MR.Const_MeshTopology._Underlying *topology, MR.Const_UndirectedEdgeBitSet._Underlying *edges, MR.VertBitSet._Underlying *store);
        return new(__MR_getIncidentVerts_3_const_MR_UndirectedEdgeBitSet_ptr(topology._UnderlyingPtr, edges is not null ? edges._UnderlyingPtr : null, store._UnderlyingPtr), is_owning: false);
    }

    /// composes the set of all vertices with all their edges in given set
    /// Generated from function `MR::getInnerVerts`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> GetInnerVerts(MR.Const_MeshTopology topology, MR.Const_UndirectedEdgeBitSet edges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getInnerVerts_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_getInnerVerts_MR_UndirectedEdgeBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_UndirectedEdgeBitSet._Underlying *edges);
        return MR.Misc.Move(new MR.VertBitSet(__MR_getInnerVerts_MR_UndirectedEdgeBitSet(topology._UnderlyingPtr, edges._UnderlyingPtr), is_owning: true));
    }
}
