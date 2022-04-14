#pragma once

#include "MRMeshTopology.h"
#include "MRMeshProject.h"
#include "MRMeshEdgePoint.h"
#include "MRUniqueThreadSafeOwner.h"
#include <cfloat>

namespace MR
{

struct [[nodiscard]] Mesh
{
    MeshTopology topology;
    VertCoords points;

    // comparison
    bool operator ==( const Mesh & b ) const { return topology == b.topology && points == b.points; }
    bool operator !=( const Mesh & b ) const { return !operator==( b ); }

    // returns coordinates of the edge origin
    Vector3f orgPnt( EdgeId e ) const { return points[ topology.org( e ) ]; }
    // returns coordinates of the edge destination
    Vector3f destPnt( EdgeId e ) const { return points[ topology.dest( e ) ]; }
    // returns a point on the edge: origin point for f=0 and destination point for f=1
    Vector3f edgePoint( EdgeId e, float f ) const { return f * destPnt( e ) + ( 1 - f ) * orgPnt( e ); }
    Vector3f edgePoint( const MeshEdgePoint & ep ) const { return edgePoint( ep.e, ep.a ); }
    // returns three points of left face of e
    MRMESH_API void getLeftTriPoints( EdgeId e, Vector3f & v0, Vector3f & v1, Vector3f & v2 ) const;
    void getLeftTriPoints( EdgeId e, Vector3f (&v)[3] ) const { getLeftTriPoints( e, v[0], v[1], v[2] ); }
    void getTriPoints( FaceId f, Vector3f & v0, Vector3f & v1, Vector3f & v2 ) const { getLeftTriPoints( topology.edgeWithLeft( f ), v0, v1, v2 ); }
    void getTriPoints( FaceId f, Vector3f (&v)[3] ) const { getTriPoints( f, v[0], v[1], v[2] ); }
    // returns interpolated coordinates of given point
    MRMESH_API Vector3f triPoint( const MeshTriPoint & p ) const;
    // returns the centroid of given triangle
    MRMESH_API Vector3f triCenter( FaceId f ) const;

    // converts face id and 3d point into barycentric representation
    MRMESH_API MeshTriPoint toTriPoint( FaceId f, const Vector3f & p ) const;
    MRMESH_API MeshTriPoint toTriPoint( const PointOnFace & p ) const;
    MRMESH_API MeshEdgePoint toEdgePoint( EdgeId e, const Vector3f & p ) const;

    // returns one of three face vertices, closest to given point
    MRMESH_API VertId getClosestVertex( const PointOnFace & p ) const;
    // returns one of three face edges, closest to given point
    MRMESH_API UndirectedEdgeId getClosestEdge( const PointOnFace & p ) const;

    // returns vector equal to edge destination point minus edge origin point
    Vector3f edgeVector( EdgeId e ) const { return destPnt( e ) - orgPnt( e ); }
    // returns Euclidean length of the edge
    float edgeLength( EdgeId e ) const { return edgeVector( e ).length(); }
    // returns squared Euclidean length of the edge (faster to compute than length)
    float edgeLengthSq( EdgeId e ) const { return edgeVector( e ).lengthSq(); }

    // computes directed double area for a triangular face from its vertices
    MRMESH_API Vector3f leftDirDblArea( EdgeId e ) const;
    Vector3f dirDblArea( FaceId f ) const { return leftDirDblArea( topology.edgeWithLeft( f ) ); }
    // returns twice the area of given face
    float dblArea( FaceId f ) const { return dirDblArea( f ).length(); }
    // returns the area of given face
    float area( FaceId f ) const { return 0.5f * dblArea( f ); }
    // returns the area of given face-region
    MRMESH_API double area( const FaceBitSet & fs ) const;
    // this version returns the area of whole mesh if argument is nullptr
    double area( const FaceBitSet * fs = nullptr ) const { return area( topology.getFaceIds( fs ) ); }
    // returns volume of closed mesh region, if region is not closed DBL_MAX is returned
    // if region is nullptr - whole mesh is region
    MRMESH_API double volume( const FaceBitSet* region = nullptr ) const;
    // computes triangular face normal from its vertices
    Vector3f leftNormal( EdgeId e ) const { return leftDirDblArea( e ).normalized(); }
    Vector3f normal( FaceId f ) const { return dirDblArea( f ).normalized(); }
    // computes sum of directed double areas of all triangles around given vertex
    MRMESH_API Vector3f dirDblArea( VertId v ) const;
    float dblArea( VertId v ) const { return dirDblArea( v ).length(); }
    // computes normal in a vertex using sum of directed areas of neighboring triangles
    Vector3f normal( VertId v ) const { return dirDblArea( v ).normalized(); }
    // computes normal in three vertices of p's triangle, then interpolates them using barycentric coordinates
    MRMESH_API Vector3f normal( const MeshTriPoint & p ) const;

    // computes pseudo-normals for signed distance calculation
    // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.9173&rep=rep1&type=pdf
    // at vertex, only region faces will be considered
    MRMESH_API Vector3f pseudonormal( VertId v, const FaceBitSet * region = nullptr ) const;
    // at edge (middle of two face normals)
    MRMESH_API Vector3f pseudonormal( EdgeId e, const FaceBitSet * region = nullptr ) const;
    // returns pseudonormal in corresponding face/edge/vertex;
    // unlike normal( const MeshTriPoint & p ), this is not a smooth function
    MRMESH_API Vector3f pseudonormal( const MeshTriPoint & p, const FaceBitSet * region = nullptr ) const;
    // given a point (p) in 3D and the closest point to in on mesh (proj), 
    // computes the signed distance from pt to mesh: positive value - outside mesh, negative - inside mesh
    MRMESH_API float signedDistance( const Vector3f & pt, const MeshTriPoint & proj, const FaceBitSet * region = nullptr ) const;
    [[deprecated]] // this version has bad precision due to PointOnFace  -> MeshTriPoint conversion
    MRMESH_API float signedDistance( const Vector3f & pt, const PointOnFace & proj, const FaceBitSet * region = nullptr ) const;
    // this version finds projection by itself in order to return signed distance from given point
    MRMESH_API float signedDistance( const Vector3f & pt ) const;
    // this version returns optional without value if the projection point is not within maxDist
    MRMESH_API std::optional<float> signedDistance( const Vector3f & pt, float maxDistSq, const FaceBitSet * region = nullptr ) const;

    // computes the sum of triangle angles at given vertex; optionally returns whether the vertex is boundary
    MRMESH_API float sumAngles( VertId v, bool * outBoundaryVert = nullptr ) const;
    // returns vertices where the sum of triangle angles is below given threshold
    MRMESH_API VertBitSet findSpikeVertices( float minSumAngle, const VertBitSet * region = nullptr ) const;

    // given an edge between two triangular faces, computes sine of dihedral angle between them:
    // 0 if both faces are in the same plane,
    // positive if the faces form convex surface,
    // negative if the faces form concave surface
    MRMESH_API float dihedralAngleSin( EdgeId e ) const;

    // given an edge between two triangular faces, computes cosine of dihedral angle between them:
    // 1 if both faces are in the same plane,
    // 0 if the surface makes right angle turn at the edge,
    // -1 if the faces overlap one another
    MRMESH_API float dihedralAngleCos( EdgeId e ) const;

    // finds all mesh edges where dihedral angle is distinct from planar PI angle on at least given value
    MRMESH_API UndirectedEdgeBitSet findCreaseEdges( float angleFromPlanar ) const;

    // computes cotangent of the angle in the left( e ) triangle opposite to e,
    // and returns 0 if left face does not exist
    MRMESH_API float leftCotan( EdgeId e ) const;
    float cotan( EdgeId e ) const { return leftCotan( e ) + leftCotan( e.sym() ); }

    // passes through all valid vertices and finds the minimal bounding box containing all of them;
    // if toWorld transformation is given then returns minimal bounding box in world space
    MRMESH_API Box3f computeBoundingBox( const AffineXf3f * toWorld = nullptr ) const;
    // returns the bounding box containing all valid vertices (implemented via getAABBTree())
    // this bounding box is insignificantly bigger that minimal box due to AABB algorithms precision
    MRMESH_API Box3f getBoundingBox() const;
    // passes through all given faces (or whole mesh if region == null) and finds the minimal bounding box containing all of them
    // if toWorld transformation is given then returns minimal bounding box in world space
    MRMESH_API Box3f computeBoundingBox( const FaceBitSet* region, const AffineXf3f* toWorld = nullptr ) const;
    // computes average length of an edge in this mesh
    MRMESH_API float averageEdgeLength() const;

    // find center location of the mesh by different means
    MRMESH_API Vector3f findCenterFromPoints() const;
    MRMESH_API Vector3f findCenterFromFaces() const;
    MRMESH_API Vector3f findCenterFromBBox() const;

    // applies given transformation to all valid mesh vertices
    MRMESH_API void transform( const AffineXf3f & xf );

    // creates new point and assigns given position to it
    MRMESH_API VertId addPoint( const Vector3f & pos );

    // append points to mesh and connect them as closed edge loop
    // returns first EdgeId of new edges
    MRMESH_API EdgeId addSeparateEdgeLoop( const std::vector<Vector3f>& contourPoints );

    // append points to mesh and connect them to given edges making edge loop
    // first point connects with first edge dest
    // last point connects with last edge org
    // note that first and last edge should have no left face
    MRMESH_API void attachEdgeLoopPart( EdgeId first, EdgeId last, const std::vector<Vector3f>& contourPoints );

    // split given edge on two equal parts, with e pointing on the second part with the same destination vertex but new origin vertex (which is returned);
    // left and right faces if valid are also subdivide by new edge each;
    // if left or right faces of the original edge were in the region, then includes new parts of these faces in the region
    MRMESH_API VertId splitEdge( EdgeId e, FaceBitSet * region = nullptr );

    // split given triangle on three triangles, introducing new vertex (which is returned) in the centroid of original triangle and connecting it to its vertices;
    // if region is given, then it must include (f) and new faces will be added there as well
    MRMESH_API VertId splitFace( FaceId f, FaceBitSet * region = nullptr );

    // appends mesh (from) in addition to this mesh: creates new edges, faces, verts and points
    MRMESH_API void addPart( const Mesh & from,
        // optionally returns mappings: from.id -> this.id
        FaceMap * outFmap = nullptr, VertMap * outVmap = nullptr, EdgeMap * outEmap = nullptr, bool rearrangeTriangles = false );
    // the same but copies only portion of (from) specified by fromFaces
    MRMESH_API void addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, const PartMapping & map = {} );
    [[deprecated]]
    MRMESH_API void addPartByMask( const Mesh & from, const FaceBitSet & fromFaces,
        // optionally returns mappings: from.id -> this.id
        FaceMap * outFmap, VertMap * outVmap = nullptr, EdgeMap * outEmap = nullptr );
    // this version has more parameters:
    //   if flipOrientation then every from triangle is inverted before adding
    MRMESH_API void addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, bool flipOrientation,
        const std::vector<std::vector<EdgeId>> & thisContours, // contours on this mesh that have to be stitched with
        const std::vector<std::vector<EdgeId>> & fromContours, // contours on from mesh during addition
        // optionally returns mappings: from.id -> this.id
        PartMapping map = {} );
    [[deprecated]]
    MRMESH_API void addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, bool flipOrientation,
        const std::vector<std::vector<EdgeId>> & thisContours, // contours on this mesh that have to be stitched with
        const std::vector<std::vector<EdgeId>> & fromContours, // contours on from mesh during addition
        // optionally returns mappings: from.id -> this.id
        FaceMap * outFmap, VertMap * outVmap = nullptr, EdgeMap * outEmap = nullptr );

    // tightly packs all arrays eliminating lone edges and invalid face, verts and points,
    // optionally returns mappings: old.id -> new.id
    MRMESH_API void pack( FaceMap * outFmap = nullptr, VertMap * outVmap = nullptr, EdgeMap * outEmap = nullptr, bool rearrangeTriangles = false );

    // All intersectRay methods are DEPRECATED! Use MRMeshIntersect.h instead
    // Intersects ray with this geometry only
    [[deprecated]]
    MRMESH_API bool intersectRay( const Vector3f& org, const Vector3f& dir, PointOnFace& res,
        float rayStart = 0.0f, float rayEnd = FLT_MAX, const FaceBitSet* region = nullptr ) const;
    [[deprecated]]
    MRMESH_API bool intersectRay( const Vector3d& org, const Vector3d& dir, PointOnFace& res,
        double rayStart = 0.0, double rayEnd = DBL_MAX, const FaceBitSet* region = nullptr ) const;
    [[deprecated]]
    MRMESH_API bool intersectRay( const Vector3f& org, const Vector3f& dir, PointOnFace& res, const AffineXf3f& rayToMeshXf,
        float rayStart = 0.0f, float rayEnd = FLT_MAX, const FaceBitSet* region = nullptr ) const;

    // finds closest point on this mesh (or its region) to given point;
    // xf is mesh-to-point transformation, if not specified then identity transformation is assumed
    MRMESH_API bool projectPoint( const Vector3f& point, PointOnFace& res, float maxDistSq = FLT_MAX, const FaceBitSet* region = nullptr, const AffineXf3f * xf = nullptr ) const;
    MRMESH_API bool projectPoint( const Vector3f& point, MeshProjectionResult& res, float maxDistSq = FLT_MAX, const FaceBitSet* region = nullptr, const AffineXf3f * xf = nullptr ) const;
    // this version returns optional without value instead of false
    MRMESH_API std::optional<MeshProjectionResult> projectPoint( const Vector3f& point, float maxDistSq = FLT_MAX, const FaceBitSet * region = nullptr, const AffineXf3f * xf = nullptr ) const;

    // returns cached aabb-tree for this mesh, creating it if it did not exist in a thread-safe manner
    MRMESH_API const AABBTree & getAABBTree() const;

    // Invalidates caches (e.g. aabb-tree) after a change in mesh geometry or topology
    MRMESH_API void invalidateCaches();

private:
    mutable UniqueThreadSafeOwner<AABBTree> AABBTreeOwner_;
};

// the purpose of this struct is to invalidate mesh cache in its destructor
struct MeshWriter
{
    Mesh & mesh;
    MeshWriter( Mesh & m ) : mesh( m ) { }
    ~MeshWriter() { mesh.invalidateCaches(); }
};

#define MR_MESH_WRITER(mesh) MR::MeshWriter _meshWriter( mesh );


} //namespace MR
