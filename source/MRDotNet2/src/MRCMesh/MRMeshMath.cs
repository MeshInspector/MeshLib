public static partial class MR
{
    /// returns coordinates of the edge origin
    /// Generated from function `MR::orgPnt`.
    public static unsafe MR.Vector3f OrgPnt(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orgPnt", ExactSpelling = true)]
        extern static MR.Vector3f __MR_orgPnt(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e);
        return __MR_orgPnt(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// returns coordinates of the edge destination
    /// Generated from function `MR::destPnt`.
    public static unsafe MR.Vector3f DestPnt(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_destPnt", ExactSpelling = true)]
        extern static MR.Vector3f __MR_destPnt(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e);
        return __MR_destPnt(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// returns vector equal to edge destination point minus edge origin point
    /// Generated from function `MR::edgeVector`.
    public static unsafe MR.Vector3f EdgeVector(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeVector", ExactSpelling = true)]
        extern static MR.Vector3f __MR_edgeVector(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e);
        return __MR_edgeVector(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// returns line segment of given edge
    /// Generated from function `MR::edgeSegment`.
    public static unsafe MR.LineSegm3f EdgeSegment_(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeSegment", ExactSpelling = true)]
        extern static MR.LineSegm3f._Underlying *__MR_edgeSegment(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e);
        return new(__MR_edgeSegment(topology._UnderlyingPtr, points._UnderlyingPtr, e), is_owning: true);
    }

    /// returns a point on the edge: origin point for f=0 and destination point for f=1
    /// Generated from function `MR::edgePoint`.
    public static unsafe MR.Vector3f EdgePoint_(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e, float f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgePoint_4", ExactSpelling = true)]
        extern static MR.Vector3f __MR_edgePoint_4(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e, float f);
        return __MR_edgePoint_4(topology._UnderlyingPtr, points._UnderlyingPtr, e, f);
    }

    /// computes coordinates of point given as edge and relative position on it
    /// Generated from function `MR::edgePoint`.
    public static unsafe MR.Vector3f EdgePoint_(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_EdgePoint ep)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgePoint_3", ExactSpelling = true)]
        extern static MR.Vector3f __MR_edgePoint_3(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_EdgePoint._Underlying *ep);
        return __MR_edgePoint_3(topology._UnderlyingPtr, points._UnderlyingPtr, ep._UnderlyingPtr);
    }

    /// computes the center of given edge
    /// Generated from function `MR::edgeCenter`.
    public static unsafe MR.Vector3f EdgeCenter(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.UndirectedEdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeCenter", ExactSpelling = true)]
        extern static MR.Vector3f __MR_edgeCenter(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.UndirectedEdgeId e);
        return __MR_edgeCenter(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// returns three points of left face of e: v0 = orgPnt( e ), v1 = destPnt( e )
    /// Generated from function `MR::getLeftTriPoints`.
    public static unsafe void GetLeftTriPoints(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e, MR.Mut_Vector3f v0, MR.Mut_Vector3f v1, MR.Mut_Vector3f v2)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getLeftTriPoints_6", ExactSpelling = true)]
        extern static void __MR_getLeftTriPoints_6(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e, MR.Mut_Vector3f._Underlying *v0, MR.Mut_Vector3f._Underlying *v1, MR.Mut_Vector3f._Underlying *v2);
        __MR_getLeftTriPoints_6(topology._UnderlyingPtr, points._UnderlyingPtr, e, v0._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
    }

    /// returns three points of left face of e: res[0] = orgPnt( e ), res[1] = destPnt( e )
    /// Generated from function `MR::getLeftTriPoints`.
    public static unsafe MR.Std.Array_MRVector3f_3 GetLeftTriPoints(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getLeftTriPoints_3", ExactSpelling = true)]
        extern static MR.Std.Array_MRVector3f_3 __MR_getLeftTriPoints_3(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e);
        return __MR_getLeftTriPoints_3(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// returns three points of given face
    /// Generated from function `MR::getTriPoints`.
    public static unsafe void GetTriPoints(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f, MR.Mut_Vector3f v0, MR.Mut_Vector3f v1, MR.Mut_Vector3f v2)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getTriPoints_6", ExactSpelling = true)]
        extern static void __MR_getTriPoints_6(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f, MR.Mut_Vector3f._Underlying *v0, MR.Mut_Vector3f._Underlying *v1, MR.Mut_Vector3f._Underlying *v2);
        __MR_getTriPoints_6(topology._UnderlyingPtr, points._UnderlyingPtr, f, v0._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
    }

    /// returns three points of given face
    /// Generated from function `MR::getTriPoints`.
    public static unsafe MR.Std.Array_MRVector3f_3 GetTriPoints(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getTriPoints_3", ExactSpelling = true)]
        extern static MR.Std.Array_MRVector3f_3 __MR_getTriPoints_3(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f);
        return __MR_getTriPoints_3(topology._UnderlyingPtr, points._UnderlyingPtr, f);
    }

    /// computes coordinates of point given as face and barycentric representation
    /// Generated from function `MR::triPoint`.
    public static unsafe MR.Vector3f TriPoint(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_MeshTriPoint p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_triPoint", ExactSpelling = true)]
        extern static MR.Vector3f __MR_triPoint(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_MeshTriPoint._Underlying *p);
        return __MR_triPoint(topology._UnderlyingPtr, points._UnderlyingPtr, p._UnderlyingPtr);
    }

    /// returns the centroid of given triangle
    /// Generated from function `MR::triCenter`.
    public static unsafe MR.Vector3f TriCenter(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_triCenter", ExactSpelling = true)]
        extern static MR.Vector3f __MR_triCenter(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f);
        return __MR_triCenter(topology._UnderlyingPtr, points._UnderlyingPtr, f);
    }

    /// returns aspect ratio of given mesh triangle equal to the ratio of the circum-radius to twice its in-radius
    /// Generated from function `MR::triangleAspectRatio`.
    public static unsafe float TriangleAspectRatio(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_triangleAspectRatio", ExactSpelling = true)]
        extern static float __MR_triangleAspectRatio(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f);
        return __MR_triangleAspectRatio(topology._UnderlyingPtr, points._UnderlyingPtr, f);
    }

    /// returns squared circumcircle diameter of given mesh triangle
    /// Generated from function `MR::circumcircleDiameterSq`.
    public static unsafe float CircumcircleDiameterSq(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_circumcircleDiameterSq", ExactSpelling = true)]
        extern static float __MR_circumcircleDiameterSq(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f);
        return __MR_circumcircleDiameterSq(topology._UnderlyingPtr, points._UnderlyingPtr, f);
    }

    /// returns circumcircle diameter of given mesh triangle
    /// Generated from function `MR::circumcircleDiameter`.
    public static unsafe float CircumcircleDiameter(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_circumcircleDiameter", ExactSpelling = true)]
        extern static float __MR_circumcircleDiameter(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f);
        return __MR_circumcircleDiameter(topology._UnderlyingPtr, points._UnderlyingPtr, f);
    }

    /// converts face id and 3d point into barycentric representation
    /// Generated from function `MR::toTriPoint`.
    public static unsafe MR.MeshTriPoint ToTriPoint(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f, MR.Const_Vector3f p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_toTriPoint_4", ExactSpelling = true)]
        extern static MR.MeshTriPoint._Underlying *__MR_toTriPoint_4(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f, MR.Const_Vector3f._Underlying *p);
        return new(__MR_toTriPoint_4(topology._UnderlyingPtr, points._UnderlyingPtr, f, p._UnderlyingPtr), is_owning: true);
    }

    /// converts face id and 3d point into barycentric representation
    /// Generated from function `MR::toTriPoint`.
    public static unsafe MR.MeshTriPoint ToTriPoint(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_PointOnFace p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_toTriPoint_3", ExactSpelling = true)]
        extern static MR.MeshTriPoint._Underlying *__MR_toTriPoint_3(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_PointOnFace._Underlying *p);
        return new(__MR_toTriPoint_3(topology._UnderlyingPtr, points._UnderlyingPtr, p._UnderlyingPtr), is_owning: true);
    }

    /// converts edge and 3d point into edge-point representation
    /// Generated from function `MR::toEdgePoint`.
    public static unsafe MR.EdgePoint ToEdgePoint(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e, MR.Const_Vector3f p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_toEdgePoint", ExactSpelling = true)]
        extern static MR.EdgePoint._Underlying *__MR_toEdgePoint(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e, MR.Const_Vector3f._Underlying *p);
        return new(__MR_toEdgePoint(topology._UnderlyingPtr, points._UnderlyingPtr, e, p._UnderlyingPtr), is_owning: true);
    }

    /// returns one of three face vertices, closest to given point
    /// Generated from function `MR::getClosestVertex`.
    public static unsafe MR.VertId GetClosestVertex(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_PointOnFace p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getClosestVertex_MR_PointOnFace", ExactSpelling = true)]
        extern static MR.VertId __MR_getClosestVertex_MR_PointOnFace(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_PointOnFace._Underlying *p);
        return __MR_getClosestVertex_MR_PointOnFace(topology._UnderlyingPtr, points._UnderlyingPtr, p._UnderlyingPtr);
    }

    /// returns one of three face vertices, closest to given point
    /// Generated from function `MR::getClosestVertex`.
    public static unsafe MR.VertId GetClosestVertex(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_MeshTriPoint p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getClosestVertex_MR_MeshTriPoint", ExactSpelling = true)]
        extern static MR.VertId __MR_getClosestVertex_MR_MeshTriPoint(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_MeshTriPoint._Underlying *p);
        return __MR_getClosestVertex_MR_MeshTriPoint(topology._UnderlyingPtr, points._UnderlyingPtr, p._UnderlyingPtr);
    }

    /// returns one of three face edges, closest to given point
    /// Generated from function `MR::getClosestEdge`.
    public static unsafe MR.UndirectedEdgeId GetClosestEdge(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_PointOnFace p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getClosestEdge_MR_PointOnFace", ExactSpelling = true)]
        extern static MR.UndirectedEdgeId __MR_getClosestEdge_MR_PointOnFace(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_PointOnFace._Underlying *p);
        return __MR_getClosestEdge_MR_PointOnFace(topology._UnderlyingPtr, points._UnderlyingPtr, p._UnderlyingPtr);
    }

    /// returns one of three face edges, closest to given point
    /// Generated from function `MR::getClosestEdge`.
    public static unsafe MR.UndirectedEdgeId GetClosestEdge(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_MeshTriPoint p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getClosestEdge_MR_MeshTriPoint", ExactSpelling = true)]
        extern static MR.UndirectedEdgeId __MR_getClosestEdge_MR_MeshTriPoint(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_MeshTriPoint._Underlying *p);
        return __MR_getClosestEdge_MR_MeshTriPoint(topology._UnderlyingPtr, points._UnderlyingPtr, p._UnderlyingPtr);
    }

    /// returns Euclidean length of the edge
    /// Generated from function `MR::edgeLength`.
    public static unsafe float EdgeLength(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.UndirectedEdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeLength", ExactSpelling = true)]
        extern static float __MR_edgeLength(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.UndirectedEdgeId e);
        return __MR_edgeLength(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// computes and returns the lengths of all edges in the mesh
    /// Generated from function `MR::edgeLengths`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeScalars> EdgeLengths(MR.Const_MeshTopology topology, MR.Const_VertCoords points)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeLengths", ExactSpelling = true)]
        extern static MR.UndirectedEdgeScalars._Underlying *__MR_edgeLengths(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points);
        return MR.Misc.Move(new MR.UndirectedEdgeScalars(__MR_edgeLengths(topology._UnderlyingPtr, points._UnderlyingPtr), is_owning: true));
    }

    /// returns squared Euclidean length of the edge (faster to compute than length)
    /// Generated from function `MR::edgeLengthSq`.
    public static unsafe float EdgeLengthSq(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.UndirectedEdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeLengthSq", ExactSpelling = true)]
        extern static float __MR_edgeLengthSq(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.UndirectedEdgeId e);
        return __MR_edgeLengthSq(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// computes directed double area of left triangular face of given edge
    /// Generated from function `MR::leftDirDblArea`.
    public static unsafe MR.Vector3f LeftDirDblArea(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_leftDirDblArea", ExactSpelling = true)]
        extern static MR.Vector3f __MR_leftDirDblArea(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e);
        return __MR_leftDirDblArea(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// computes directed double area for a triangular face from its vertices
    /// Generated from function `MR::dirDblArea`.
    public static unsafe MR.Vector3f DirDblArea(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dirDblArea_MR_FaceId", ExactSpelling = true)]
        extern static MR.Vector3f __MR_dirDblArea_MR_FaceId(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f);
        return __MR_dirDblArea_MR_FaceId(topology._UnderlyingPtr, points._UnderlyingPtr, f);
    }

    /// computes and returns the directed double area for every (region) vertex in the mesh
    /// Generated from function `MR::dirDblAreas`.
    public static unsafe MR.Misc._Moved<MR.VertCoords> DirDblAreas(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dirDblAreas", ExactSpelling = true)]
        extern static MR.VertCoords._Underlying *__MR_dirDblAreas(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *region);
        return MR.Misc.Move(new MR.VertCoords(__MR_dirDblAreas(topology._UnderlyingPtr, points._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
    }

    /// returns twice the area of given face
    /// Generated from function `MR::dblArea`.
    public static unsafe float DblArea(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dblArea_MR_FaceId", ExactSpelling = true)]
        extern static float __MR_dblArea_MR_FaceId(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f);
        return __MR_dblArea_MR_FaceId(topology._UnderlyingPtr, points._UnderlyingPtr, f);
    }

    /// returns the area of given face
    /// Generated from function `MR::area`.
    public static unsafe float Area(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_area_MR_FaceId", ExactSpelling = true)]
        extern static float __MR_area_MR_FaceId(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f);
        return __MR_area_MR_FaceId(topology._UnderlyingPtr, points._UnderlyingPtr, f);
    }

    /// computes the area of given face-region (or whole mesh)
    /// Generated from function `MR::area`.
    public static unsafe double Area(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_FaceBitSet? fs = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_area_const_MR_FaceBitSet_ptr", ExactSpelling = true)]
        extern static double __MR_area_const_MR_FaceBitSet_ptr(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_FaceBitSet._Underlying *fs);
        return __MR_area_const_MR_FaceBitSet_ptr(topology._UnderlyingPtr, points._UnderlyingPtr, fs is not null ? fs._UnderlyingPtr : null);
    }

    /// computes the sum of directed areas for faces from given region (or whole mesh)
    /// Generated from function `MR::dirArea`.
    public static unsafe MR.Vector3d DirArea(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_FaceBitSet? fs = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dirArea", ExactSpelling = true)]
        extern static MR.Vector3d __MR_dirArea(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_FaceBitSet._Underlying *fs);
        return __MR_dirArea(topology._UnderlyingPtr, points._UnderlyingPtr, fs is not null ? fs._UnderlyingPtr : null);
    }

    /// computes the sum of absolute projected area of faces from given region (or whole mesh) as visible if look from given direction
    /// Generated from function `MR::projArea`.
    public static unsafe double ProjArea(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_Vector3f dir, MR.Const_FaceBitSet? fs = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_projArea", ExactSpelling = true)]
        extern static double __MR_projArea(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_Vector3f._Underlying *dir, MR.Const_FaceBitSet._Underlying *fs);
        return __MR_projArea(topology._UnderlyingPtr, points._UnderlyingPtr, dir._UnderlyingPtr, fs is not null ? fs._UnderlyingPtr : null);
    }

    /// returns volume of the object surrounded by given region (or whole mesh if (region) is nullptr);
    /// if the region has holes then each hole will be virtually filled by adding triangles for each edge and the hole's geometrical center
    /// Generated from function `MR::volume`.
    public static unsafe double Volume(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_volume", ExactSpelling = true)]
        extern static double __MR_volume(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_FaceBitSet._Underlying *region);
        return __MR_volume(topology._UnderlyingPtr, points._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
    }

    /// computes the perimeter of the hole specified by one of its edges with no valid left face (left is hole)
    /// Generated from function `MR::holePerimiter`.
    public static unsafe double HolePerimiter(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_holePerimiter", ExactSpelling = true)]
        extern static double __MR_holePerimiter(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e);
        return __MR_holePerimiter(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// computes directed area of the hole specified by one of its edges with no valid left face (left is hole);
    /// if the hole is planar then returned vector is orthogonal to the plane pointing outside and its magnitude is equal to hole area
    /// Generated from function `MR::holeDirArea`.
    public static unsafe MR.Vector3d HoleDirArea(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_holeDirArea", ExactSpelling = true)]
        extern static MR.Vector3d __MR_holeDirArea(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e);
        return __MR_holeDirArea(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// computes unit vector that is both orthogonal to given edge and to the normal of its left triangle, the vector is directed inside left triangle
    /// Generated from function `MR::leftTangent`.
    public static unsafe MR.Vector3f LeftTangent(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_leftTangent", ExactSpelling = true)]
        extern static MR.Vector3f __MR_leftTangent(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e);
        return __MR_leftTangent(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// computes triangular face normal from its vertices
    /// Generated from function `MR::leftNormal`.
    public static unsafe MR.Vector3f LeftNormal(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_leftNormal", ExactSpelling = true)]
        extern static MR.Vector3f __MR_leftNormal(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e);
        return __MR_leftNormal(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// computes triangular face normal from its vertices
    /// Generated from function `MR::normal`.
    public static unsafe MR.Vector3f Normal(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_normal_MR_FaceId", ExactSpelling = true)]
        extern static MR.Vector3f __MR_normal_MR_FaceId(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f);
        return __MR_normal_MR_FaceId(topology._UnderlyingPtr, points._UnderlyingPtr, f);
    }

    /// returns the plane containing given triangular face with normal looking outwards
    /// Generated from function `MR::getPlane3f`.
    public static unsafe MR.Plane3f GetPlane3f(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getPlane3f", ExactSpelling = true)]
        extern static MR.Plane3f._Underlying *__MR_getPlane3f(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f);
        return new(__MR_getPlane3f(topology._UnderlyingPtr, points._UnderlyingPtr, f), is_owning: true);
    }

    /// Generated from function `MR::getPlane3d`.
    public static unsafe MR.Plane3d GetPlane3d(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getPlane3d", ExactSpelling = true)]
        extern static MR.Plane3d._Underlying *__MR_getPlane3d(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceId f);
        return new(__MR_getPlane3d(topology._UnderlyingPtr, points._UnderlyingPtr, f), is_owning: true);
    }

    /// computes sum of directed double areas of all triangles around given vertex
    /// Generated from function `MR::dirDblArea`.
    public static unsafe MR.Vector3f DirDblArea(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.VertId v)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dirDblArea_MR_VertId", ExactSpelling = true)]
        extern static MR.Vector3f __MR_dirDblArea_MR_VertId(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.VertId v);
        return __MR_dirDblArea_MR_VertId(topology._UnderlyingPtr, points._UnderlyingPtr, v);
    }

    /// computes the length of summed directed double areas of all triangles around given vertex
    /// Generated from function `MR::dblArea`.
    public static unsafe float DblArea(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.VertId v)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dblArea_MR_VertId", ExactSpelling = true)]
        extern static float __MR_dblArea_MR_VertId(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.VertId v);
        return __MR_dblArea_MR_VertId(topology._UnderlyingPtr, points._UnderlyingPtr, v);
    }

    /// computes normal in a vertex using sum of directed areas of neighboring triangles
    /// Generated from function `MR::normal`.
    public static unsafe MR.Vector3f Normal(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.VertId v)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_normal_MR_VertId", ExactSpelling = true)]
        extern static MR.Vector3f __MR_normal_MR_VertId(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.VertId v);
        return __MR_normal_MR_VertId(topology._UnderlyingPtr, points._UnderlyingPtr, v);
    }

    /// computes normal in three vertices of p's triangle, then interpolates them using barycentric coordinates and normalizes again;
    /// this is the same normal as in rendering with smooth shading
    /// Generated from function `MR::normal`.
    public static unsafe MR.Vector3f Normal(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_MeshTriPoint p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_normal_MR_MeshTriPoint", ExactSpelling = true)]
        extern static MR.Vector3f __MR_normal_MR_MeshTriPoint(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_MeshTriPoint._Underlying *p);
        return __MR_normal_MR_MeshTriPoint(topology._UnderlyingPtr, points._UnderlyingPtr, p._UnderlyingPtr);
    }

    /// computes angle-weighted sum of normals of incident faces of given vertex (only (region) faces will be considered);
    /// the sum is normalized before returning
    /// Generated from function `MR::pseudonormal`.
    public static unsafe MR.Vector3f Pseudonormal(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.VertId v, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pseudonormal_MR_VertId", ExactSpelling = true)]
        extern static MR.Vector3f __MR_pseudonormal_MR_VertId(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.VertId v, MR.Const_FaceBitSet._Underlying *region);
        return __MR_pseudonormal_MR_VertId(topology._UnderlyingPtr, points._UnderlyingPtr, v, region is not null ? region._UnderlyingPtr : null);
    }

    /// computes normalized half sum of face normals sharing given edge (only (region) faces will be considered);
    /// Generated from function `MR::pseudonormal`.
    public static unsafe MR.Vector3f Pseudonormal(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.UndirectedEdgeId e, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pseudonormal_MR_UndirectedEdgeId", ExactSpelling = true)]
        extern static MR.Vector3f __MR_pseudonormal_MR_UndirectedEdgeId(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.UndirectedEdgeId e, MR.Const_FaceBitSet._Underlying *region);
        return __MR_pseudonormal_MR_UndirectedEdgeId(topology._UnderlyingPtr, points._UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null);
    }

    /// returns pseudonormal in corresponding face/edge/vertex for signed distance calculation
    /// as suggested in the article "Signed Distance Computation Using the Angle Weighted Pseudonormal" by J. Andreas Baerentzen and Henrik Aanaes,
    /// https://backend.orbit.dtu.dk/ws/portalfiles/portal/3977815/B_rentzen.pdf
    /// unlike normal( const MeshTriPoint & p ), this is not a smooth function
    /// Generated from function `MR::pseudonormal`.
    public static unsafe MR.Vector3f Pseudonormal(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_MeshTriPoint p, MR.Const_FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pseudonormal_MR_MeshTriPoint", ExactSpelling = true)]
        extern static MR.Vector3f __MR_pseudonormal_MR_MeshTriPoint(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_MeshTriPoint._Underlying *p, MR.Const_FaceBitSet._Underlying *region);
        return __MR_pseudonormal_MR_MeshTriPoint(topology._UnderlyingPtr, points._UnderlyingPtr, p._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
    }

    /// computes the sum of triangle angles at given vertex; optionally returns whether the vertex is on boundary
    /// Generated from function `MR::sumAngles`.
    public static unsafe float SumAngles(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.VertId v, MR.Misc.InOut<bool>? outBoundaryVert = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sumAngles", ExactSpelling = true)]
        extern static float __MR_sumAngles(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.VertId v, bool *outBoundaryVert);
        bool __value_outBoundaryVert = outBoundaryVert is not null ? outBoundaryVert.Value : default(bool);
        var __ret = __MR_sumAngles(topology._UnderlyingPtr, points._UnderlyingPtr, v, outBoundaryVert is not null ? &__value_outBoundaryVert : null);
        if (outBoundaryVert is not null) outBoundaryVert.Value = __value_outBoundaryVert;
        return __ret;
    }

    /// returns vertices where the sum of triangle angles is below given threshold
    /// Generated from function `MR::findSpikeVertices`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRVertBitSet_StdString> FindSpikeVertices(MR.Const_MeshTopology topology, MR.Const_VertCoords points, float minSumAngle, MR.Const_VertBitSet? region = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSpikeVertices", ExactSpelling = true)]
        extern static MR.Expected_MRVertBitSet_StdString._Underlying *__MR_findSpikeVertices(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, float minSumAngle, MR.Const_VertBitSet._Underlying *region, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRVertBitSet_StdString(__MR_findSpikeVertices(topology._UnderlyingPtr, points._UnderlyingPtr, minSumAngle, region is not null ? region._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }

    /// given an edge between two triangular faces, computes sine of dihedral angle between them:
    /// 0 if both faces are in the same plane,
    /// positive if the faces form convex surface,
    /// negative if the faces form concave surface
    /// Generated from function `MR::dihedralAngleSin`.
    public static unsafe float DihedralAngleSin(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.UndirectedEdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dihedralAngleSin", ExactSpelling = true)]
        extern static float __MR_dihedralAngleSin(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.UndirectedEdgeId e);
        return __MR_dihedralAngleSin(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// given an edge between two triangular faces, computes cosine of dihedral angle between them:
    /// 1 if both faces are in the same plane,
    /// 0 if the surface makes right angle turn at the edge,
    /// -1 if the faces overlap one another
    /// Generated from function `MR::dihedralAngleCos`.
    public static unsafe float DihedralAngleCos(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.UndirectedEdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dihedralAngleCos", ExactSpelling = true)]
        extern static float __MR_dihedralAngleCos(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.UndirectedEdgeId e);
        return __MR_dihedralAngleCos(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// given an edge between two triangular faces, computes the dihedral angle between them:
    /// 0 if both faces are in the same plane,
    /// positive if the faces form convex surface,
    /// negative if the faces form concave surface;
    /// please consider the usage of faster dihedralAngleSin(e) and dihedralAngleCos(e)
    /// Generated from function `MR::dihedralAngle`.
    public static unsafe float DihedralAngle(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.UndirectedEdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dihedralAngle", ExactSpelling = true)]
        extern static float __MR_dihedralAngle(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.UndirectedEdgeId e);
        return __MR_dihedralAngle(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// computes discrete mean curvature in given vertex, measures in length^-1;
    /// 0 for planar regions, positive for convex surface, negative for concave surface
    /// Generated from function `MR::discreteMeanCurvature`.
    public static unsafe float DiscreteMeanCurvature(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.VertId v)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_discreteMeanCurvature_MR_VertId", ExactSpelling = true)]
        extern static float __MR_discreteMeanCurvature_MR_VertId(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.VertId v);
        return __MR_discreteMeanCurvature_MR_VertId(topology._UnderlyingPtr, points._UnderlyingPtr, v);
    }

    /// computes discrete mean curvature in given edge, measures in length^-1;
    /// 0 for planar regions, positive for convex surface, negative for concave surface
    /// Generated from function `MR::discreteMeanCurvature`.
    public static unsafe float DiscreteMeanCurvature(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.UndirectedEdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_discreteMeanCurvature_MR_UndirectedEdgeId", ExactSpelling = true)]
        extern static float __MR_discreteMeanCurvature_MR_UndirectedEdgeId(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.UndirectedEdgeId e);
        return __MR_discreteMeanCurvature_MR_UndirectedEdgeId(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// computes discrete Gaussian curvature (or angle defect) at given vertex,
    /// which 0 in inner vertices on planar mesh parts and reaches 2*pi on needle's tip, see http://math.uchicago.edu/~may/REU2015/REUPapers/Upadhyay.pdf
    /// optionally returns whether the vertex is on boundary
    /// Generated from function `MR::discreteGaussianCurvature`.
    public static unsafe float DiscreteGaussianCurvature(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.VertId v, MR.Misc.InOut<bool>? outBoundaryVert = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_discreteGaussianCurvature", ExactSpelling = true)]
        extern static float __MR_discreteGaussianCurvature(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.VertId v, bool *outBoundaryVert);
        bool __value_outBoundaryVert = outBoundaryVert is not null ? outBoundaryVert.Value : default(bool);
        var __ret = __MR_discreteGaussianCurvature(topology._UnderlyingPtr, points._UnderlyingPtr, v, outBoundaryVert is not null ? &__value_outBoundaryVert : null);
        if (outBoundaryVert is not null) outBoundaryVert.Value = __value_outBoundaryVert;
        return __ret;
    }

    /// finds all mesh edges where dihedral angle is distinct from planar PI angle on at least given value
    /// Generated from function `MR::findCreaseEdges`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> FindCreaseEdges(MR.Const_MeshTopology topology, MR.Const_VertCoords points, float angleFromPlanar)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCreaseEdges", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_findCreaseEdges(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, float angleFromPlanar);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_findCreaseEdges(topology._UnderlyingPtr, points._UnderlyingPtr, angleFromPlanar), is_owning: true));
    }

    /// computes cotangent of the angle in the left( e ) triangle opposite to e,
    /// and returns 0 if left face does not exist
    /// Generated from function `MR::leftCotan`.
    public static unsafe float LeftCotan(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_leftCotan", ExactSpelling = true)]
        extern static float __MR_leftCotan(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e);
        return __MR_leftCotan(topology._UnderlyingPtr, points._UnderlyingPtr, e);
    }

    /// computes sum of cotangents of the angle in the left and right triangles opposite to given edge,
    /// consider cotangents zero for not existing triangles
    /// Generated from function `MR::cotan`.
    public static unsafe float Cotan(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.UndirectedEdgeId ue)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_cotan", ExactSpelling = true)]
        extern static float __MR_cotan(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.UndirectedEdgeId ue);
        return __MR_cotan(topology._UnderlyingPtr, points._UnderlyingPtr, ue);
    }

    /// computes quadratic form in the vertex as the sum of squared distances from
    /// 1) planes of adjacent triangles, with the weight equal to the angle of adjacent triangle at this vertex divided on PI in case of angleWeigted=true;
    /// 2) lines of adjacent boundary and crease edges
    /// Generated from function `MR::quadraticForm`.
    public static unsafe MR.QuadraticForm3f QuadraticForm(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.VertId v, bool angleWeigted, MR.Const_FaceBitSet? region = null, MR.Const_UndirectedEdgeBitSet? creases = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_quadraticForm", ExactSpelling = true)]
        extern static MR.QuadraticForm3f._Underlying *__MR_quadraticForm(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.VertId v, byte angleWeigted, MR.Const_FaceBitSet._Underlying *region, MR.Const_UndirectedEdgeBitSet._Underlying *creases);
        return new(__MR_quadraticForm(topology._UnderlyingPtr, points._UnderlyingPtr, v, angleWeigted ? (byte)1 : (byte)0, region is not null ? region._UnderlyingPtr : null, creases is not null ? creases._UnderlyingPtr : null), is_owning: true);
    }

    /// passes through all given faces (or whole mesh if region == null) and finds the minimal bounding box containing all of them
    /// if toWorld transformation is given then returns minimal bounding box in world space
    /// Generated from function `MR::computeBoundingBox`.
    public static unsafe MR.Box3f ComputeBoundingBox(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.Const_FaceBitSet? region, MR.Const_AffineXf3f? toWorld = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeBoundingBox", ExactSpelling = true)]
        extern static MR.Box3f __MR_computeBoundingBox(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_FaceBitSet._Underlying *region, MR.Const_AffineXf3f._Underlying *toWorld);
        return __MR_computeBoundingBox(topology._UnderlyingPtr, points._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, toWorld is not null ? toWorld._UnderlyingPtr : null);
    }

    /// computes average length of an edge in the mesh given by (topology, points)
    /// Generated from function `MR::averageEdgeLength`.
    public static unsafe float AverageEdgeLength(MR.Const_MeshTopology topology, MR.Const_VertCoords points)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_averageEdgeLength", ExactSpelling = true)]
        extern static float __MR_averageEdgeLength(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points);
        return __MR_averageEdgeLength(topology._UnderlyingPtr, points._UnderlyingPtr);
    }

    /// computes average position of all valid mesh vertices
    /// Generated from function `MR::findCenterFromPoints`.
    public static unsafe MR.Vector3f FindCenterFromPoints(MR.Const_MeshTopology topology, MR.Const_VertCoords points)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCenterFromPoints", ExactSpelling = true)]
        extern static MR.Vector3f __MR_findCenterFromPoints(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points);
        return __MR_findCenterFromPoints(topology._UnderlyingPtr, points._UnderlyingPtr);
    }

    /// computes center of mass considering that density of all triangles is the same
    /// Generated from function `MR::findCenterFromFaces`.
    public static unsafe MR.Vector3f FindCenterFromFaces(MR.Const_MeshTopology topology, MR.Const_VertCoords points)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCenterFromFaces", ExactSpelling = true)]
        extern static MR.Vector3f __MR_findCenterFromFaces(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points);
        return __MR_findCenterFromFaces(topology._UnderlyingPtr, points._UnderlyingPtr);
    }

    /// computes bounding box and returns its center
    /// Generated from function `MR::findCenterFromBBox`.
    public static unsafe MR.Vector3f FindCenterFromBBox(MR.Const_MeshTopology topology, MR.Const_VertCoords points)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCenterFromBBox", ExactSpelling = true)]
        extern static MR.Vector3f __MR_findCenterFromBBox(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points);
        return __MR_findCenterFromBBox(topology._UnderlyingPtr, points._UnderlyingPtr);
    }
}
