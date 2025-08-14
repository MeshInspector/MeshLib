#pragma once

#include "MRPch/MRBindingMacros.h"
#include "MRMeshMath.h"
#include "MRMeshBuilderTypes.h"
#include "MRMeshProject.h"
#include "MREdgePoint.h"
#include "MRSharedThreadSafeOwner.h"
#include "MRWriter.h"
#include "MRConstants.h"
#include "MRProgressCallback.h"
#include <cfloat>

namespace MR
{

/// \defgroup MeshGroup Mesh

/// This class represents a mesh, including topology (connectivity) information and point coordinates,
/// as well as some caches to accelerate search algorithms
/// \ingroup MeshGroup
struct [[nodiscard]] Mesh
{
    MeshTopology topology;
    VertCoords points;

    /// construct mesh from vertex coordinates and a set of triangles with given ids
    [[nodiscard]] MRMESH_API static Mesh fromTriangles(
        VertCoords vertexCoordinates,
        const Triangulation& t, const MeshBuilder::BuildSettings& settings = {}, ProgressCallback cb = {} );

    /// construct mesh from TriMesh representation
    [[nodiscard]] MRMESH_API static Mesh fromTriMesh(
        TriMesh && triMesh, ///< points of triMesh will be moves in the result
        const MeshBuilder::BuildSettings& settings = {}, ProgressCallback cb = {} );

    /// construct mesh from vertex coordinates and a set of triangles with given ids;
    /// unlike simple fromTriangles() it tries to resolve non-manifold vertices by creating duplicate vertices
    [[nodiscard]] MRMESH_API static Mesh fromTrianglesDuplicatingNonManifoldVertices(
        VertCoords vertexCoordinates,
        Triangulation & t,
        std::vector<MeshBuilder::VertDuplication> * dups = nullptr,
        const MeshBuilder::BuildSettings & settings = {} );

    /// construct mesh from vertex coordinates and construct mesh topology from face soup,
    /// where each face can have arbitrary degree (not only triangles);
    /// all non-triangular faces will be automatically subdivided on triangles
    [[nodiscard]] MRMESH_API static Mesh fromFaceSoup(
        VertCoords vertexCoordinates,
        const std::vector<VertId> & verts, const Vector<MeshBuilder::VertSpan, FaceId> & faces,
        const MeshBuilder::BuildSettings& settings = {}, ProgressCallback cb = {} );

    /// construct mesh from point triples;
    /// \param duplicateNonManifoldVertices = false, all coinciding points are given the same VertId in the result;
    /// \param duplicateNonManifoldVertices = true, it tries to avoid non-manifold vertices by creating duplicate vertices with same coordinates
    [[nodiscard]] MRMESH_API static Mesh fromPointTriples( const std::vector<Triangle3f> & posTriples, bool duplicateNonManifoldVertices );

    /// compare that two meshes are exactly the same
    [[nodiscard]] MRMESH_API bool operator ==( const Mesh & b ) const;

    /// returns coordinates of the edge origin
    [[nodiscard]] Vector3f orgPnt( EdgeId e ) const { return MR::orgPnt( topology, points, e ); }

    /// returns coordinates of the edge destination
    [[nodiscard]] Vector3f destPnt( EdgeId e ) const { return MR::destPnt( topology, points, e ); }

    /// returns vector equal to edge destination point minus edge origin point
    [[nodiscard]] Vector3f edgeVector( EdgeId e ) const { return MR::edgeVector( topology, points, e ); }

    /// returns line segment of given edge
    [[nodiscard]] MRMESH_API LineSegm3f edgeSegment( EdgeId e ) const;

    /// returns a point on the edge: origin point for f=0 and destination point for f=1
    [[nodiscard]] Vector3f edgePoint( EdgeId e, float f ) const { return MR::edgePoint( topology, points, e, f ); }

    /// computes coordinates of point given as edge and relative position on it
    [[nodiscard]] Vector3f edgePoint( const MeshEdgePoint & ep ) const { return MR::edgePoint( topology, points, ep ); }

    /// computes the center of given edge
    [[nodiscard]] Vector3f edgeCenter( UndirectedEdgeId e ) const { return MR::edgeCenter( topology, points, e ); }

    /// returns three points of left face of e: v0 = orgPnt( e ), v1 = destPnt( e )
    void getLeftTriPoints( EdgeId e, Vector3f & v0, Vector3f & v1, Vector3f & v2 ) const { return MR::getLeftTriPoints( topology, points, e, v0, v1, v2 ); }

    /// returns three points of left face of e: v[0] = orgPnt( e ), v[1] = destPnt( e )
    /// This one is not in the bindings because of the reference-to-array parameter.
    MR_BIND_IGNORE void getLeftTriPoints( EdgeId e, Vector3f (&v)[3] ) const { return MR::getLeftTriPoints( topology, points, e, v ); }

    /// returns three points of left face of e: res[0] = orgPnt( e ), res[1] = destPnt( e )
    [[nodiscard]] Triangle3f getLeftTriPoints( EdgeId e ) const { return MR::getLeftTriPoints( topology, points, e ); }

    /// returns three points of given face
    void getTriPoints( FaceId f, Vector3f & v0, Vector3f & v1, Vector3f & v2 ) const { return MR::getTriPoints( topology, points, f, v0, v1, v2 ); }

    /// returns three points of given face
    /// This one is not in the bindings because of the reference-to-array parameter.
    MR_BIND_IGNORE void getTriPoints( FaceId f, Vector3f (&v)[3] ) const { return MR::getTriPoints( topology, points, f, v ); }

    /// returns three points of given face
    [[nodiscard]] Triangle3f getTriPoints( FaceId f ) const { return MR::getTriPoints( topology, points, f ); }

    /// computes coordinates of point given as face and barycentric representation
    [[nodiscard]] Vector3f triPoint( const MeshTriPoint & p ) const { return MR::triPoint( topology, points, p ); }

    /// returns the centroid of given triangle
    [[nodiscard]] Vector3f triCenter( FaceId f ) const { return MR::triCenter( topology, points, f ); }

    /// returns aspect ratio of given mesh triangle equal to the ratio of the circum-radius to twice its in-radius
    [[nodiscard]] float triangleAspectRatio( FaceId f ) const { return MR::triangleAspectRatio( topology, points, f ); }

    /// returns squared circumcircle diameter of given mesh triangle
    [[nodiscard]] float circumcircleDiameterSq( FaceId f ) const { return MR::circumcircleDiameterSq( topology, points, f ); }

    /// returns circumcircle diameter of given mesh triangle
    [[nodiscard]] float circumcircleDiameter( FaceId f ) const { return MR::circumcircleDiameter( topology, points, f ); }

    /// converts vertex into barycentric representation
    [[nodiscard]] MRMESH_API MeshTriPoint toTriPoint( VertId v ) const;

    /// converts face id and 3d point into barycentric representation
    [[nodiscard]] MeshTriPoint toTriPoint( FaceId f, const Vector3f & p ) const { return MR::toTriPoint( topology, points, f, p ); }

    /// converts face id and 3d point into barycentric representation
    [[nodiscard]] MeshTriPoint toTriPoint( const PointOnFace& p ) const { return MR::toTriPoint( topology, points, p ); }

    /// converts vertex into edge-point representation
    [[nodiscard]] MRMESH_API MeshEdgePoint toEdgePoint( VertId v ) const;

    /// converts edge and 3d point into edge-point representation
    [[nodiscard]] MeshEdgePoint toEdgePoint( EdgeId e, const Vector3f & p ) const { return MR::toEdgePoint( topology, points, e, p ); }

    /// returns one of three face vertices, closest to given point
    [[nodiscard]] VertId getClosestVertex( const PointOnFace & p ) const { return MR::getClosestVertex( topology, points, p ); }

    /// returns one of three face vertices, closest to given point
    [[nodiscard]] VertId getClosestVertex( const MeshTriPoint & p ) const { return MR::getClosestVertex( topology, points, p ); }

    /// returns one of three face edges, closest to given point
    [[nodiscard]] UndirectedEdgeId getClosestEdge( const PointOnFace & p ) const { return MR::getClosestEdge( topology, points, p ); }

    /// returns one of three face edges, closest to given point
    [[nodiscard]] UndirectedEdgeId getClosestEdge( const MeshTriPoint & p ) const { return MR::getClosestEdge( topology, points, p ); }

    /// returns Euclidean length of the edge
    [[nodiscard]] float edgeLength( UndirectedEdgeId e ) const { return MR::edgeLength( topology, points, e ); }

    /// computes and returns the lengths of all edges in the mesh
    [[nodiscard]] UndirectedEdgeScalars edgeLengths() const { return MR::edgeLengths( topology, points ); }

    /// returns squared Euclidean length of the edge (faster to compute than length)
    [[nodiscard]] float edgeLengthSq( UndirectedEdgeId e ) const { return MR::edgeLengthSq( topology, points, e ); }

    /// computes directed double area of left triangular face of given edge
    [[nodiscard]]  Vector3f leftDirDblArea( EdgeId e ) const { return MR::leftDirDblArea( topology, points, e ); }

    /// computes directed double area for a triangular face from its vertices
    [[nodiscard]] Vector3f dirDblArea( FaceId f ) const { return MR::dirDblArea( topology, points, f ); }

    /// returns twice the area of given face
    [[nodiscard]] float dblArea( FaceId f ) const { return MR::dblArea( topology, points, f ); }

    /// returns the area of given face
    [[nodiscard]] float area( FaceId f ) const { return MR::area( topology, points, f ); }

    /// computes the area of given face-region
    [[nodiscard]] double area( const FaceBitSet & fs ) const { return MR::area( topology, points, fs ); }

    /// computes the area of given face-region (or whole mesh)
    [[nodiscard]] double area( const FaceBitSet * fs = nullptr ) const { return MR::area( topology, points, fs ); }

    /// computes the sum of directed areas for faces from given region
    [[nodiscard]] Vector3d dirArea( const FaceBitSet & fs ) const { return MR::dirArea( topology, points, fs ); }

    /// computes the sum of directed areas for faces from given region (or whole mesh)
    [[nodiscard]] Vector3d dirArea( const FaceBitSet * fs = nullptr ) const { return MR::dirArea( topology, points, fs ); }

    /// computes the sum of absolute projected area of faces from given region as visible if look from given direction
    [[nodiscard]] double projArea( const Vector3f & dir, const FaceBitSet & fs ) const { return MR::projArea( topology, points, dir, fs ); }

    /// computes the sum of absolute projected area of faces from given region (or whole mesh) as visible if look from given direction
    [[nodiscard]] double projArea( const Vector3f & dir, const FaceBitSet * fs = nullptr ) const { return MR::projArea( topology, points, dir, fs ); }

    /// returns volume of the object surrounded by given region (or whole mesh if (region) is nullptr);
    /// if the region has holes then each hole will be virtually filled by adding triangles for each edge and the hole's geometrical center
    [[nodiscard]] double volume( const FaceBitSet* region = nullptr ) const { return MR::volume( topology, points, region ); }

    /// computes the perimeter of the hole specified by one of its edges with no valid left face (left is hole)
    [[nodiscard]] double holePerimiter( EdgeId e ) const { return MR::holePerimiter( topology, points, e ); }

    /// computes directed area of the hole specified by one of its edges with no valid left face (left is hole);
    /// if the hole is planar then returned vector is orthogonal to the plane pointing outside and its magnitude is equal to hole area
    [[nodiscard]] Vector3d holeDirArea( EdgeId e ) const { return MR::holeDirArea( topology, points, e ); }

    /// computes unit vector that is both orthogonal to given edge and to the normal of its left triangle, the vector is directed inside left triangle
    [[nodiscard]] Vector3f leftTangent( EdgeId e ) const { return MR::leftTangent( topology, points, e ); }

    /// computes triangular face normal from its vertices
    [[nodiscard]] Vector3f leftNormal( EdgeId e ) const { return MR::leftNormal( topology, points, e ); }

    /// computes triangular face normal from its vertices
    [[nodiscard]] Vector3f normal( FaceId f ) const { return MR::normal( topology, points, f ); }

    /// returns the plane containing given triangular face with normal looking outwards
    [[nodiscard]] MRMESH_API Plane3f getPlane3f( FaceId f ) const;
    [[nodiscard]] MRMESH_API Plane3d getPlane3d( FaceId f ) const;

    /// computes sum of directed double areas of all triangles around given vertex
    [[nodiscard]] Vector3f dirDblArea( VertId v ) const { return MR::dirDblArea( topology, points, v ); }

    /// computes the length of summed directed double areas of all triangles around given vertex
    [[nodiscard]] float dblArea( VertId v ) const { return MR::dblArea( topology, points, v ); }

    /// computes normal in a vertex using sum of directed areas of neighboring triangles
    [[nodiscard]] Vector3f normal( VertId v ) const { return MR::normal( topology, points, v ); }

    /// computes normal in three vertices of p's triangle, then interpolates them using barycentric coordinates and normalizes again;
    /// this is the same normal as in rendering with smooth shading
    [[nodiscard]] Vector3f normal( const MeshTriPoint & p ) const { return MR::normal( topology, points, p ); }

    /// computes angle-weighted sum of normals of incident faces of given vertex (only (region) faces will be considered);
    /// the sum is normalized before returning
    [[nodiscard]] Vector3f pseudonormal( VertId v, const FaceBitSet * region = nullptr ) const { return MR::pseudonormal( topology, points, v, region ); }

    /// computes normalized half sum of face normals sharing given edge (only (region) faces will be considered);
    [[nodiscard]] Vector3f pseudonormal( UndirectedEdgeId e, const FaceBitSet * region = nullptr ) const { return MR::pseudonormal( topology, points, e, region ); }

    /// returns pseudonormal in corresponding face/edge/vertex for signed distance calculation
    /// as suggested in the article "Signed Distance Computation Using the Angle Weighted Pseudonormal" by J. Andreas Baerentzen and Henrik Aanaes,
    /// https://backend.orbit.dtu.dk/ws/portalfiles/portal/3977815/B_rentzen.pdf
    /// unlike normal( const MeshTriPoint & p ), this is not a smooth function
    [[nodiscard]] Vector3f pseudonormal( const MeshTriPoint & p, const FaceBitSet * region = nullptr ) const { return MR::pseudonormal( topology, points, p, region ); }

    /// given a point (pt) in 3D and the closest point to in on mesh (proj),
    /// \return signed distance from pt to mesh: positive value - outside mesh, negative - inside mesh;
    /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
    [[nodiscard]] MRMESH_API float signedDistance( const Vector3f & pt, const MeshProjectionResult & proj, const FaceBitSet * region = nullptr ) const;
    [[deprecated]] MRMESH_API MR_BIND_IGNORE float signedDistance( const Vector3f & pt, const MeshTriPoint & proj, const FaceBitSet * region = nullptr ) const;

    /// given a point (pt) in 3D, computes the closest point on mesh, and
    /// \return signed distance from pt to mesh: positive value - outside mesh, negative - inside mesh;
    /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
    [[nodiscard]] MRMESH_API float signedDistance( const Vector3f & pt ) const;

    /// given a point (pt) in 3D, computes the closest point on mesh, and
    /// \return signed distance from pt to mesh: positive value - outside mesh, negative - inside mesh;
    ///   or std::nullopt if the projection point is not within maxDist;
    /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
    [[nodiscard]] MRMESH_API std::optional<float> signedDistance( const Vector3f & pt, float maxDistSq, const FaceBitSet * region = nullptr ) const;

    /// computes generalized winding number in a point (pt), which is
    /// * for closed mesh with normals outside: 1 inside, 0 outside;
    /// * for planar mesh: 0.5 inside, -0.5 outside;
    /// and in general is equal to (portion of solid angle where inside part of mesh is observable) minus (portion of solid angle where outside part of mesh is observable)
    /// \param beta determines the precision of fast approximation: the more the better, recommended value 2 or more
    [[nodiscard]] MRMESH_API float calcFastWindingNumber( const Vector3f & pt, float beta = 2 ) const;

    /// computes whether a point (pt) is located outside the object surrounded by this mesh using generalized winding number
    /// \param beta determines the precision of winding number computation: the more the better, recommended value 2 or more
    [[nodiscard]] bool isOutside( const Vector3f & pt, float windingNumberThreshold = 0.5f, float beta = 2 ) const { return calcFastWindingNumber( pt, beta ) <= windingNumberThreshold; }

    /// computes whether a point (pt) is located outside the object surrounded by this mesh
    /// using pseudonormal at the closest point to in on mesh (proj);
    /// this method works much faster than \ref isOutside but can return wrong sign if the closest point is located on self-intersecting part of the mesh
    [[nodiscard]] MRMESH_API bool isOutsideByProjNorm( const Vector3f & pt, const MeshProjectionResult & proj, const FaceBitSet * region = nullptr ) const;

    /// computes the sum of triangle angles at given vertex; optionally returns whether the vertex is on boundary
    [[nodiscard]] float sumAngles( VertId v, bool * outBoundaryVert = nullptr ) const { return MR::sumAngles( topology, points, v, outBoundaryVert ); }

    /// returns vertices where the sum of triangle angles is below given threshold
    [[nodiscard]] Expected<VertBitSet> findSpikeVertices( float minSumAngle, const VertBitSet* region = nullptr, const ProgressCallback& cb = {} ) const { return MR::findSpikeVertices( topology, points, minSumAngle, region, cb ); }

    /// given an edge between two triangular faces, computes sine of dihedral angle between them:
    /// 0 if both faces are in the same plane,
    /// positive if the faces form convex surface,
    /// negative if the faces form concave surface
    [[nodiscard]] float dihedralAngleSin( UndirectedEdgeId e ) const { return MR::dihedralAngleSin( topology, points, e ); }

    /// given an edge between two triangular faces, computes cosine of dihedral angle between them:
    /// 1 if both faces are in the same plane,
    /// 0 if the surface makes right angle turn at the edge,
    /// -1 if the faces overlap one another
    [[nodiscard]] float dihedralAngleCos( UndirectedEdgeId e ) const { return MR::dihedralAngleCos( topology, points, e ); }

    /// given an edge between two triangular faces, computes the dihedral angle between them:
    /// 0 if both faces are in the same plane,
    /// positive if the faces form convex surface,
    /// negative if the faces form concave surface;
    /// please consider the usage of faster dihedralAngleSin(e) and dihedralAngleCos(e)
    [[nodiscard]] float dihedralAngle( UndirectedEdgeId e ) const { return MR::dihedralAngle( topology, points, e ); }

    /// computes discrete mean curvature in given vertex, measures in length^-1;
    /// 0 for planar regions, positive for convex surface, negative for concave surface
    [[nodiscard]] float discreteMeanCurvature( VertId v ) const { return MR::discreteMeanCurvature( topology, points, v ); }

    /// computes discrete mean curvature in given edge, measures in length^-1;
    /// 0 for planar regions, positive for convex surface, negative for concave surface
    [[nodiscard]] float discreteMeanCurvature( UndirectedEdgeId e ) const { return MR::discreteMeanCurvature( topology, points, e ); }

    /// computes discrete Gaussian curvature (or angle defect) at given vertex,
    /// which 0 in inner vertices on planar mesh parts and reaches 2*pi on needle's tip, see http://math.uchicago.edu/~may/REU2015/REUPapers/Upadhyay.pdf
    /// optionally returns whether the vertex is on boundary
    [[nodiscard]] float discreteGaussianCurvature( VertId v, bool * outBoundaryVert = nullptr ) const { return MR::discreteGaussianCurvature( topology, points, v, outBoundaryVert ); }

    /// finds all mesh edges where dihedral angle is distinct from planar PI angle on at least given value
    [[nodiscard]] UndirectedEdgeBitSet findCreaseEdges( float angleFromPlanar ) const { return MR::findCreaseEdges( topology, points, angleFromPlanar ); }

    /// computes cotangent of the angle in the left( e ) triangle opposite to e,
    /// and returns 0 if left face does not exist
    [[nodiscard]] float leftCotan( EdgeId e ) const { return MR::leftCotan( topology, points, e ); }

    /// computes sum of cotangents of the angle in the left and right triangles opposite to given edge,
    /// consider cotangents zero for not existing triangles
    [[nodiscard]] float cotan( UndirectedEdgeId ue ) const { return MR::cotan( topology, points, ue ); }

    /// computes quadratic form in the vertex as the sum of squared distances from
    /// 1) planes of adjacent triangles, with the weight equal to the angle of adjacent triangle at this vertex divided on PI in case of angleWeigted=true;
    /// 2) lines of adjacent boundary and crease edges
    [[nodiscard]] MRMESH_API QuadraticForm3f quadraticForm( VertId v, bool angleWeigted,
        const FaceBitSet * region = nullptr, const UndirectedEdgeBitSet * creases = nullptr ) const;

    /// returns the bounding box containing all valid vertices (implemented via getAABBTree())
    /// this bounding box is insignificantly bigger that minimal box due to AABB algorithms precision
    [[nodiscard]] MRMESH_API Box3f getBoundingBox() const;

    /// passes through all valid vertices and finds the minimal bounding box containing all of them;
    /// if toWorld transformation is given then returns minimal bounding box in world space
    [[nodiscard]] MRMESH_API Box3f computeBoundingBox( const AffineXf3f * toWorld = nullptr ) const;

    /// passes through all given faces (or whole mesh if region == null) and finds the minimal bounding box containing all of them
    /// if toWorld transformation is given then returns minimal bounding box in world space
    [[nodiscard]] MRMESH_API Box3f computeBoundingBox( const FaceBitSet* region, const AffineXf3f* toWorld = nullptr ) const;

    /// computes average length of an edge in this mesh
    [[nodiscard]] float averageEdgeLength() const { return MR::averageEdgeLength( topology, points ); }

    /// computes average position of all valid mesh vertices
    [[nodiscard]] Vector3f findCenterFromPoints() const { return MR::findCenterFromPoints( topology, points ); }

    /// computes center of mass considering that density of all triangles is the same
    [[nodiscard]] Vector3f findCenterFromFaces() const { return MR::findCenterFromFaces( topology, points ); }

    /// computes bounding box and returns its center
    [[nodiscard]] Vector3f findCenterFromBBox() const { return MR::findCenterFromBBox( topology, points ); }

    /// for all points not in topology.getValidVerts() sets coordinates to (0,0,0)
    MRMESH_API void zeroUnusedPoints();

    /// applies given transformation to specified vertices
    /// if region is nullptr, all valid mesh vertices are used
    /// \snippet cpp-examples/MeshModification.dox.cpp MeshTransform
    MRMESH_API void transform( const AffineXf3f& xf, const VertBitSet* region = nullptr );

    /// creates new point and assigns given position to it
    MRMESH_API VertId addPoint( const Vector3f & pos );

    /// append points to mesh and connect them as closed edge loop
    /// returns first EdgeId of new edges
    MRMESH_API EdgeId addSeparateEdgeLoop(const std::vector<Vector3f>& contourPoints);

    /// append points to mesh and connect them
    /// returns first EdgeId of new edges
    MRMESH_API EdgeId addSeparateContours( const Contours3f& contours, const AffineXf3f* xf = nullptr );

    /// append points to mesh and connect them to given edges making edge loop
    /// first point connects with first edge dest
    /// last point connects with last edge org
    /// note that first and last edge should have no left face
    MRMESH_API void attachEdgeLoopPart( EdgeId first, EdgeId last, const std::vector<Vector3f>& contourPoints );

    /// split given edge on two parts:
    /// dest(returned-edge) = org(e) - newly created vertex,
    /// org(returned-edge) = org(e-before-split),
    /// dest(e) = dest(e-before-split)
    /// \details left and right faces of given edge if valid are also subdivided on two parts each;
    /// the split edge will keep both face IDs and their degrees, and the new edge will have new face IDs and new faces are triangular;
    /// if left or right faces of the original edge were in the region, then include new parts of these faces in the region
    /// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
    MRMESH_API EdgeId splitEdge( EdgeId e, const Vector3f & newVertPos, FaceBitSet * region = nullptr, FaceHashMap * new2Old = nullptr );
    // same, but split given edge on two equal parts
    EdgeId splitEdge( EdgeId e, FaceBitSet * region = nullptr, FaceHashMap * new2Old = nullptr ) { return splitEdge( e, edgeCenter( e ), region, new2Old ); }

    /// split given triangle on three triangles, introducing new vertex with given coordinates and connecting it to original triangle vertices;
    /// if region is given, then it must include (f) and new faces will be added there as well
    /// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
    MRMESH_API VertId splitFace( FaceId f, const Vector3f & newVertPos, FaceBitSet * region = nullptr, FaceHashMap * new2Old = nullptr );
    // same, putting new vertex in the centroid of original triangle
    VertId splitFace( FaceId f, FaceBitSet * region = nullptr, FaceHashMap * new2Old = nullptr ) { return splitFace( f, triCenter( f ), region, new2Old ); }

    /// appends another mesh as separate connected component(s) to this
    MRMESH_API void addMesh( const Mesh & from, PartMapping map = {}, bool rearrangeTriangles = false );
    MRMESH_API void addMesh( const Mesh & from,
        // optionally returns mappings: from.id -> this.id
        FaceMap * outFmap, VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr, bool rearrangeTriangles = false );
    [[deprecated]] MR_BIND_IGNORE void addPart( const Mesh & from, FaceMap * outFmap = nullptr, VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr, bool rearrangeTriangles = false )
        { addMesh( from, outFmap, outVmap, outEmap, rearrangeTriangles ); }

    /// appends whole or part of another mesh as separate connected component(s) to this
    MRMESH_API void addMeshPart( const MeshPart & from, const PartMapping & map );
    [[deprecated]] MR_BIND_IGNORE void addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, const PartMapping & map ) { addMeshPart( { from, &fromFaces }, map ); }

    /// appends whole or part of another mesh to this joining added faces with existed ones along given contours
    /// \param flipOrientation true means that every (from) triangle is inverted before adding
    MRMESH_API void addMeshPart( const MeshPart & from, bool flipOrientation = false,
        const std::vector<EdgePath> & thisContours = {}, // contours on this mesh that have to be stitched with
        const std::vector<EdgePath> & fromContours = {}, // contours on from mesh during addition
        // optionally returns mappings: from.id -> this.id
        PartMapping map = {} );
    [[deprecated]] MR_BIND_IGNORE void addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, bool flipOrientation = false,
        const std::vector<EdgePath> & thisContours = {}, const std::vector<EdgePath> & fromContours = {}, const PartMapping & map = {} )
        { addMeshPart( { from, &fromFaces }, flipOrientation, thisContours, fromContours, map ); }

    /// creates new mesh from given triangles of this mesh
    MRMESH_API Mesh cloneRegion( const FaceBitSet & region, bool flipOrientation = false, const PartMapping & map = {} ) const;

    /// tightly packs all arrays eliminating lone edges and invalid faces, vertices and points
    MRMESH_API void pack( const PartMapping & map = {}, bool rearrangeTriangles = false );
    MRMESH_API void pack( /// optionally returns mappings: old.id -> new.id
        FaceMap * outFmap, VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr, bool rearrangeTriangles = false );

    /// tightly packs all arrays eliminating lone edges and invalid faces, vertices and points,
    /// reorder all faces, vertices and edges according to given maps, each containing old id -> new id mapping
    MRMESH_API Expected<void> pack( const PackMapping & map, ProgressCallback cb = {} );

    /// packs tightly and rearranges vertices, triangles and edges to put close in space elements in close indices
    /// \param preserveAABBTree whether to keep valid mesh's AABB tree after return (it will take longer to compute and it will occupy more memory)
    MRMESH_API PackMapping packOptimally( bool preserveAABBTree = true );
    MRMESH_API Expected<PackMapping> packOptimally( bool preserveAABBTree, ProgressCallback cb );

    /// deletes multiple given faces, also deletes adjacent edges and vertices if they were not shared by remaining faces and not in \param keepFaces
    MRMESH_API void deleteFaces( const FaceBitSet & fs, const UndirectedEdgeBitSet * keepEdges = nullptr );

    /// finds the closest mesh point on this mesh (or its region) to given point;
    /// \param point source location to look the closest to
    /// \param res found closest point including Euclidean coordinates and FaceId
    /// \param maxDistSq search only in the ball with sqrt(maxDistSq) radius around given point, smaller value here increases performance
    /// \param xf is mesh-to-point transformation, if not specified then identity transformation is assumed and works much faster;
    /// \return false if no mesh point is found in the ball with sqrt(maxDistSq) radius around given point
    [[nodiscard]] MRMESH_API bool projectPoint( const Vector3f& point, PointOnFace& res, float maxDistSq = FLT_MAX, const FaceBitSet* region = nullptr, const AffineXf3f * xf = nullptr ) const;

    /// finds the closest mesh point on this mesh (or its region) to given point;
    /// \param point source location to look the closest to
    /// \param res found closest point including Euclidean coordinates, barycentric coordinates, FaceId and squared distance to point
    /// \param maxDistSq search only in the ball with sqrt(maxDistSq) radius around given point, smaller value here increases performance
    /// \param xf is mesh-to-point transformation, if not specified then identity transformation is assumed and works much faster;
    /// \return false if no mesh point is found in the ball with sqrt(maxDistSq) radius around given point
    [[nodiscard]] MRMESH_API bool projectPoint( const Vector3f& point, MeshProjectionResult& res, float maxDistSq = FLT_MAX, const FaceBitSet* region = nullptr, const AffineXf3f * xf = nullptr ) const;
    [[nodiscard]] bool findClosestPoint( const Vector3f& point, MeshProjectionResult& res, float maxDistSq = FLT_MAX, const FaceBitSet* region = nullptr, const AffineXf3f * xf = nullptr ) const { return projectPoint( point, res, maxDistSq, region, xf ); }

    /// finds the closest mesh point on this mesh (or its region) to given point;
    /// \param point source location to look the closest to
    /// \param maxDistSq search only in the ball with sqrt(maxDistSq) radius around given point, smaller value here increases performance
    /// \param xf is mesh-to-point transformation, if not specified then identity transformation is assumed and works much faster;
    /// \return found closest point including Euclidean coordinates, barycentric coordinates, FaceId and squared distance to point
    ///         or std::nullopt if no mesh point is found in the ball with sqrt(maxDistSq) radius around given point
    [[nodiscard]] MRMESH_API MeshProjectionResult projectPoint( const Vector3f& point, float maxDistSq = FLT_MAX, const FaceBitSet * region = nullptr, const AffineXf3f * xf = nullptr ) const;
    [[nodiscard]] MeshProjectionResult findClosestPoint( const Vector3f& point, float maxDistSq = FLT_MAX, const FaceBitSet * region = nullptr, const AffineXf3f * xf = nullptr ) const { return projectPoint( point, maxDistSq, region, xf ); }

    /// returns cached aabb-tree for this mesh, creating it if it did not exist in a thread-safe manner
    MRMESH_API const AABBTree & getAABBTree() const;

    /// returns cached aabb-tree for this mesh, but does not create it if it did not exist
    [[nodiscard]] const AABBTree * getAABBTreeNotCreate() const { return AABBTreeOwner_.get(); }

    /// returns cached aabb-tree for points of this mesh, creating it if it did not exist in a thread-safe manner
    MRMESH_API const AABBTreePoints & getAABBTreePoints() const;

    /// returns cached aabb-tree for points of this mesh, but does not create it if it did not exist
    [[nodiscard]] const AABBTreePoints * getAABBTreePointsNotCreate() const { return AABBTreePointsOwner_.get(); }

    /// returns cached dipoles of aabb-tree nodes for this mesh, creating it if it did not exist in a thread-safe manner
    MRMESH_API const Dipoles & getDipoles() const;

    /// returns cached dipoles of aabb-tree nodes for this mesh, but does not create it if it did not exist
    [[nodiscard]] const Dipoles * getDipolesNotCreate() const { return dipolesOwner_.get(); }

    /// invalidates caches (aabb-trees) after any change in mesh geometry or topology
    /// \param pointsChanged specifies whether points have changed (otherwise only topology has changed)
    MRMESH_API void invalidateCaches( bool pointsChanged = true );

    /// updates existing caches in case of few vertices were changed insignificantly,
    /// and topology remained unchanged;
    /// it shall be considered as a faster alternative to invalidateCaches() and following rebuild of trees
    MRMESH_API void updateCaches( const VertBitSet & changedVerts );

    // returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

    /// requests the removal of unused capacity
    MRMESH_API void shrinkToFit();

    /// reflects the mesh from a given plane
    MRMESH_API void mirror( const Plane3f& plane );

private:
    mutable SharedThreadSafeOwner<AABBTree> AABBTreeOwner_;
    mutable SharedThreadSafeOwner<AABBTreePoints> AABBTreePointsOwner_;
    mutable SharedThreadSafeOwner<Dipoles> dipolesOwner_;
};

} //namespace MR
