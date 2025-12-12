public static partial class MR
{
    /// extracts all iso-lines from given scalar field and iso-value=0
    /// Generated from function `MR::extractIsolines`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgePoint> ExtractIsolines(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMRVertId vertValues, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extractIsolines_3", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgePoint._Underlying *__MR_extractIsolines_3(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMRVertId._Underlying *vertValues, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgePoint(__MR_extractIsolines_3(topology._UnderlyingPtr, vertValues._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
    }

    /// quickly returns true if extractIsolines produce not-empty set for the same arguments
    /// Generated from function `MR::hasAnyIsoline`.
    public static unsafe bool HasAnyIsoline(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMRVertId vertValues, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_hasAnyIsoline_3", ExactSpelling = true)]
        extern static byte __MR_hasAnyIsoline_3(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMRVertId._Underlying *vertValues, MR.Const_FaceBitSet._Underlying *region);
        return __MR_hasAnyIsoline_3(topology._UnderlyingPtr, vertValues._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null) != 0;
    }

    /// extracts all iso-lines from given scalar field and iso-value
    /// Generated from function `MR::extractIsolines`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgePoint> ExtractIsolines(MR.Const_MeshTopology topology, MR.Const_VertScalars vertValues, float isoValue, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extractIsolines_4", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgePoint._Underlying *__MR_extractIsolines_4(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertScalars._Underlying *vertValues, float isoValue, MR.Const_FaceBitSet._Underlying *region);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgePoint(__MR_extractIsolines_4(topology._UnderlyingPtr, vertValues._UnderlyingPtr, isoValue, region is not null ? region._UnderlyingPtr : null), is_owning: true));
    }

    /// quickly returns true if extractIsolines produce not-empty set for the same arguments
    /// Generated from function `MR::hasAnyIsoline`.
    public static unsafe bool HasAnyIsoline(MR.Const_MeshTopology topology, MR.Const_VertScalars vertValues, float isoValue, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_hasAnyIsoline_4", ExactSpelling = true)]
        extern static byte __MR_hasAnyIsoline_4(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertScalars._Underlying *vertValues, float isoValue, MR.Const_FaceBitSet._Underlying *region);
        return __MR_hasAnyIsoline_4(topology._UnderlyingPtr, vertValues._UnderlyingPtr, isoValue, region is not null ? region._UnderlyingPtr : null) != 0;
    }

    /// extracts all plane sections of given mesh
    /// Generated from function `MR::extractPlaneSections`.
    /// Parameter `u` defaults to `UseAABBTree::Yes`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgePoint> ExtractPlaneSections(MR.Const_MeshPart mp, MR.Const_Plane3f plane, MR.UseAABBTree? u = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extractPlaneSections", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgePoint._Underlying *__MR_extractPlaneSections(MR.Const_MeshPart._Underlying *mp, MR.Const_Plane3f._Underlying *plane, MR.UseAABBTree *u);
        MR.UseAABBTree __deref_u = u.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgePoint(__MR_extractPlaneSections(mp._UnderlyingPtr, plane._UnderlyingPtr, u.HasValue ? &__deref_u : null), is_owning: true));
    }

    /// quickly returns true if extractPlaneSections produce not-empty set for the same arguments
    /// Generated from function `MR::hasAnyPlaneSection`.
    /// Parameter `u` defaults to `UseAABBTree::Yes`.
    public static unsafe bool HasAnyPlaneSection(MR.Const_MeshPart mp, MR.Const_Plane3f plane, MR.UseAABBTree? u = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_hasAnyPlaneSection", ExactSpelling = true)]
        extern static byte __MR_hasAnyPlaneSection(MR.Const_MeshPart._Underlying *mp, MR.Const_Plane3f._Underlying *plane, MR.UseAABBTree *u);
        MR.UseAABBTree __deref_u = u.GetValueOrDefault();
        return __MR_hasAnyPlaneSection(mp._UnderlyingPtr, plane._UnderlyingPtr, u.HasValue ? &__deref_u : null) != 0;
    }

    /// extracts all sections of given mesh with the plane z=zLevel
    /// Generated from function `MR::extractXYPlaneSections`.
    /// Parameter `u` defaults to `UseAABBTree::Yes`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgePoint> ExtractXYPlaneSections(MR.Const_MeshPart mp, float zLevel, MR.UseAABBTree? u = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extractXYPlaneSections", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgePoint._Underlying *__MR_extractXYPlaneSections(MR.Const_MeshPart._Underlying *mp, float zLevel, MR.UseAABBTree *u);
        MR.UseAABBTree __deref_u = u.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgePoint(__MR_extractXYPlaneSections(mp._UnderlyingPtr, zLevel, u.HasValue ? &__deref_u : null), is_owning: true));
    }

    /// quickly returns true if extractXYPlaneSections produce not-empty set for the same arguments
    /// Generated from function `MR::hasAnyXYPlaneSection`.
    /// Parameter `u` defaults to `UseAABBTree::Yes`.
    public static unsafe bool HasAnyXYPlaneSection(MR.Const_MeshPart mp, float zLevel, MR.UseAABBTree? u = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_hasAnyXYPlaneSection", ExactSpelling = true)]
        extern static byte __MR_hasAnyXYPlaneSection(MR.Const_MeshPart._Underlying *mp, float zLevel, MR.UseAABBTree *u);
        MR.UseAABBTree __deref_u = u.GetValueOrDefault();
        return __MR_hasAnyXYPlaneSection(mp._UnderlyingPtr, zLevel, u.HasValue ? &__deref_u : null) != 0;
    }

    /// finds all intersected triangles by the plane z=zLevel
    /// \return the section's line segment within each such triangle;
    /// \param faces optional output of the same size as return, where for each line segment one can find its triangle's id
    /// \details this function must be faster than
    /// extractXYPlaneSections function when connecting continuous contours take most of the time
    /// Generated from function `MR::findTriangleSectionsByXYPlane`.
    /// Parameter `u` defaults to `UseAABBTree::Yes`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRLineSegm3f> FindTriangleSectionsByXYPlane(MR.Const_MeshPart mp, float zLevel, MR.Std.Vector_MRFaceId? faces = null, MR.UseAABBTree? u = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTriangleSectionsByXYPlane", ExactSpelling = true)]
        extern static MR.Std.Vector_MRLineSegm3f._Underlying *__MR_findTriangleSectionsByXYPlane(MR.Const_MeshPart._Underlying *mp, float zLevel, MR.Std.Vector_MRFaceId._Underlying *faces, MR.UseAABBTree *u);
        MR.UseAABBTree __deref_u = u.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MRLineSegm3f(__MR_findTriangleSectionsByXYPlane(mp._UnderlyingPtr, zLevel, faces is not null ? faces._UnderlyingPtr : null, u.HasValue ? &__deref_u : null), is_owning: true));
    }

    /// track section of plane set by start point, direction and surface normal in start point 
    /// in given direction while given distance or
    /// mesh boundary is not reached, or track looped
    /// negative distance means moving in opposite direction
    /// returns track on surface and end point (same as start if path has looped)
    /// Generated from function `MR::trackSection`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgePoint> TrackSection(MR.Const_MeshPart mp, MR.Const_MeshTriPoint start, MR.MeshTriPoint end, MR.Const_Vector3f direction, float distance)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_trackSection_MR_MeshTriPoint_ref", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgePoint._Underlying *__MR_trackSection_MR_MeshTriPoint_ref(MR.Const_MeshPart._Underlying *mp, MR.Const_MeshTriPoint._Underlying *start, MR.MeshTriPoint._Underlying *end, MR.Const_Vector3f._Underlying *direction, float distance);
        return MR.Misc.Move(new MR.Std.Vector_MREdgePoint(__MR_trackSection_MR_MeshTriPoint_ref(mp._UnderlyingPtr, start._UnderlyingPtr, end._UnderlyingPtr, direction._UnderlyingPtr, distance), is_owning: true));
    }

    /// track section of plane set by start point, end point and planePoint
    /// from start to end
    /// \param ccw - if true use start->end->planePoint plane, otherwise use start->planePoint->end (changes direction of plane tracking)
    /// returns track on surface without end point (return error if path was looped or reached boundary)
    /// Generated from function `MR::trackSection`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMREdgePoint_StdString> TrackSection(MR.Const_MeshPart mp, MR.Const_MeshTriPoint start, MR.Const_MeshTriPoint end, MR.Const_Vector3f planePoint, bool ccw)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_trackSection_const_MR_MeshTriPoint_ref", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMREdgePoint_StdString._Underlying *__MR_trackSection_const_MR_MeshTriPoint_ref(MR.Const_MeshPart._Underlying *mp, MR.Const_MeshTriPoint._Underlying *start, MR.Const_MeshTriPoint._Underlying *end, MR.Const_Vector3f._Underlying *planePoint, byte ccw);
        return MR.Misc.Move(new MR.Expected_StdVectorMREdgePoint_StdString(__MR_trackSection_const_MR_MeshTriPoint_ref(mp._UnderlyingPtr, start._UnderlyingPtr, end._UnderlyingPtr, planePoint._UnderlyingPtr, ccw ? (byte)1 : (byte)0), is_owning: true));
    }

    /// returns true if left(isoline[i].e) == right(isoline[i+1].e) and valid for all i;
    /// all above functions produce consistently oriented lines
    /// Generated from function `MR::isConsistentlyOriented`.
    public static unsafe bool IsConsistentlyOriented(MR.Const_MeshTopology topology, MR.Std.Const_Vector_MREdgePoint isoline)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isConsistentlyOriented", ExactSpelling = true)]
        extern static byte __MR_isConsistentlyOriented(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgePoint._Underlying *isoline);
        return __MR_isConsistentlyOriented(topology._UnderlyingPtr, isoline._UnderlyingPtr) != 0;
    }

    /// for a consistently oriented isoline, returns all faces it goes inside
    /// Generated from function `MR::getCrossedFaces`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetCrossedFaces(MR.Const_MeshTopology topology, MR.Std.Const_Vector_MREdgePoint isoline)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getCrossedFaces", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_getCrossedFaces(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgePoint._Underlying *isoline);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_getCrossedFaces(topology._UnderlyingPtr, isoline._UnderlyingPtr), is_owning: true));
    }

    /// converts PlaneSections in 2D contours by computing coordinate of each point, applying given xf to it, and retaining only x and y
    /// Generated from function `MR::planeSectionToContour2f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector2f> PlaneSectionToContour2f(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgePoint section, MR.Const_AffineXf3f meshToPlane)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_planeSectionToContour2f", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector2f._Underlying *__MR_planeSectionToContour2f(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgePoint._Underlying *section, MR.Const_AffineXf3f._Underlying *meshToPlane);
        return MR.Misc.Move(new MR.Std.Vector_MRVector2f(__MR_planeSectionToContour2f(mesh._UnderlyingPtr, section._UnderlyingPtr, meshToPlane._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::planeSectionsToContours2f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2f> PlaneSectionsToContours2f(MR.Const_Mesh mesh, MR.Std.Const_Vector_StdVectorMREdgePoint sections, MR.Const_AffineXf3f meshToPlane)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_planeSectionsToContours2f", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector2f._Underlying *__MR_planeSectionsToContours2f(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_StdVectorMREdgePoint._Underlying *sections, MR.Const_AffineXf3f._Underlying *meshToPlane);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2f(__MR_planeSectionsToContours2f(mesh._UnderlyingPtr, sections._UnderlyingPtr, meshToPlane._UnderlyingPtr), is_owning: true));
    }
}
