#pragma once

#include "MRPch/MRBindingMacros.h"
#include "MRVector.h"
#include "MRMeshTopology.h"
#include "MRPointOnFace.h"

namespace MR
{

/// returns coordinates of the edge origin
[[nodiscard]] inline Vector3f orgPnt( const MeshTopology & topology, const VertCoords & points, EdgeId e )
{
    return points[ topology.org( e ) ];
}

/// returns coordinates of the edge destination
[[nodiscard]] inline Vector3f destPnt( const MeshTopology & topology, const VertCoords & points, EdgeId e )
{
    return points[ topology.dest( e ) ];
}

/// returns vector equal to edge destination point minus edge origin point
[[nodiscard]] inline Vector3f edgeVector( const MeshTopology & topology, const VertCoords & points, EdgeId e )
{
    return destPnt( topology, points, e ) - orgPnt( topology, points, e );
}

/// returns line segment of given edge
MRMESH_API LineSegm3f edgeSegment( const MeshTopology & topology, const VertCoords & points, EdgeId e );

/// returns a point on the edge: origin point for f=0 and destination point for f=1
[[nodiscard]] inline Vector3f edgePoint( const MeshTopology & topology, const VertCoords & points, EdgeId e, float f )
{
    return f * destPnt( topology, points, e ) + ( 1 - f ) * orgPnt( topology, points, e );
}

/// computes coordinates of point given as edge and relative position on it
[[nodiscard]] inline Vector3f edgePoint( const MeshTopology & topology, const VertCoords & points, const MeshEdgePoint & ep )
{
    return edgePoint( topology, points, ep.e, ep.a );
}

/// computes the center of given edge
[[nodiscard]] inline Vector3f edgeCenter( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId e )
{
    return edgePoint( topology, points, e, 0.5f );
}

/// returns three points of left face of e: v0 = orgPnt( e ), v1 = destPnt( e )
MRMESH_API void getLeftTriPoints( const MeshTopology & topology, const VertCoords & points, EdgeId e, Vector3f & v0, Vector3f & v1, Vector3f & v2 );

/// returns three points of left face of e: v[0] = orgPnt( e ), v[1] = destPnt( e )
/// This one is not in the bindings because of the reference-to-array parameter.
MR_BIND_IGNORE inline void getLeftTriPoints( const MeshTopology & topology, const VertCoords & points, EdgeId e, Vector3f (&v)[3] )
{
    getLeftTriPoints( topology, points, e, v[0], v[1], v[2] );
}

/// returns three points of left face of e: res[0] = orgPnt( e ), res[1] = destPnt( e )
[[nodiscard]] inline Triangle3f getLeftTriPoints( const MeshTopology & topology, const VertCoords & points, EdgeId e )
{
    Triangle3f res;
    getLeftTriPoints( topology, points, e, res[0], res[1], res[2] );
    return res;
}

/// returns three points of given face
inline void getTriPoints( const MeshTopology & topology, const VertCoords & points, FaceId f, Vector3f & v0, Vector3f & v1, Vector3f & v2 )
{
    getLeftTriPoints( topology, points, topology.edgeWithLeft( f ), v0, v1, v2 );
}

/// returns three points of given face
/// This one is not in the bindings because of the reference-to-array parameter.
MR_BIND_IGNORE inline void getTriPoints( const MeshTopology & topology, const VertCoords & points, FaceId f, Vector3f (&v)[3] )
{
    getTriPoints( topology, points, f, v[0], v[1], v[2] );
}

/// returns three points of given face
[[nodiscard]] inline Triangle3f getTriPoints( const MeshTopology & topology, const VertCoords & points, FaceId f )
{
    Triangle3f res;
    getTriPoints( topology, points, f, res[0], res[1], res[2] );
    return res;
}

/// computes coordinates of point given as face and barycentric representation
[[nodiscard]] MRMESH_API Vector3f triPoint( const MeshTopology & topology, const VertCoords & points, const MeshTriPoint & p );

/// returns the centroid of given triangle
[[nodiscard]] MRMESH_API Vector3f triCenter( const MeshTopology & topology, const VertCoords & points, FaceId f );

/// returns aspect ratio of given mesh triangle equal to the ratio of the circum-radius to twice its in-radius
[[nodiscard]] MRMESH_API float triangleAspectRatio( const MeshTopology & topology, const VertCoords & points, FaceId f );

/// returns squared circumcircle diameter of given mesh triangle
[[nodiscard]] MRMESH_API float circumcircleDiameterSq( const MeshTopology & topology, const VertCoords & points, FaceId f );

/// returns circumcircle diameter of given mesh triangle
[[nodiscard]] MRMESH_API float circumcircleDiameter( const MeshTopology & topology, const VertCoords & points, FaceId f );

/// converts face id and 3d point into barycentric representation
[[nodiscard]] MRMESH_API MeshTriPoint toTriPoint( const MeshTopology & topology, const VertCoords & points, FaceId f, const Vector3f & p );

/// converts face id and 3d point into barycentric representation
[[nodiscard]] MRMESH_API MeshTriPoint toTriPoint( const MeshTopology & topology, const VertCoords & points, const PointOnFace& p );

/// converts edge and 3d point into edge-point representation
[[nodiscard]] MRMESH_API MeshEdgePoint toEdgePoint( const MeshTopology & topology, const VertCoords & points, EdgeId e, const Vector3f & p );

/// returns one of three face vertices, closest to given point
[[nodiscard]] MRMESH_API VertId getClosestVertex( const MeshTopology & topology, const VertCoords & points, const PointOnFace & p );

/// returns one of three face vertices, closest to given point
[[nodiscard]] inline VertId getClosestVertex( const MeshTopology & topology, const VertCoords & points, const MeshTriPoint & p )
{
    return getClosestVertex( topology, points, PointOnFace{ topology.left( p.e ), triPoint( topology, points, p ) } );
}

/// returns one of three face edges, closest to given point
[[nodiscard]] MRMESH_API UndirectedEdgeId getClosestEdge( const MeshTopology & topology, const VertCoords & points, const PointOnFace & p );

/// returns one of three face edges, closest to given point
[[nodiscard]] inline UndirectedEdgeId getClosestEdge( const MeshTopology & topology, const VertCoords & points, const MeshTriPoint & p )
{
    return getClosestEdge( topology, points, PointOnFace{ topology.left( p.e ), triPoint( topology, points, p ) } );
}

/// returns Euclidean length of the edge
[[nodiscard]] inline float edgeLength( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId e )
{
    return edgeVector( topology, points, e ).length();
}

/// computes and returns the lengths of all edges in the mesh
[[nodiscard]] MRMESH_API UndirectedEdgeScalars edgeLengths( const MeshTopology & topology, const VertCoords & points );

/// returns squared Euclidean length of the edge (faster to compute than length)
[[nodiscard]] inline float edgeLengthSq( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId e )
{
    return edgeVector( topology, points, e ).lengthSq();
}

/// computes directed double area of left triangular face of given edge
[[nodiscard]] MRMESH_API Vector3f leftDirDblArea( const MeshTopology & topology, const VertCoords & points, EdgeId e );

/// computes directed double area for a triangular face from its vertices
[[nodiscard]] inline Vector3f dirDblArea( const MeshTopology & topology, const VertCoords & points, FaceId f )
{
    return leftDirDblArea( topology, points, topology.edgeWithLeft( f ) );
}

/// computes and returns the directed double area for every (region) vertex in the mesh
[[nodiscard]] MRMESH_API Vector<Vector3f, VertId> dirDblAreas( const MeshTopology & topology, const VertCoords & points, const VertBitSet * region = nullptr );

/// returns twice the area of given face
[[nodiscard]] inline float dblArea( const MeshTopology & topology, const VertCoords & points, FaceId f )
{
    return dirDblArea( topology, points, f ).length();
}

/// returns the area of given face
[[nodiscard]] inline float area( const MeshTopology & topology, const VertCoords & points, FaceId f )
{
    return 0.5f * dblArea( topology, points, f );
}

/// computes the area of given face-region
[[nodiscard]] MRMESH_API double area( const MeshTopology & topology, const VertCoords & points, const FaceBitSet & fs );

/// computes the area of given face-region (or whole mesh)
[[nodiscard]] inline double area( const MeshTopology & topology, const VertCoords & points, const FaceBitSet * fs = nullptr )
{
    return area( topology, points, topology.getFaceIds( fs ) );
}

/// computes the sum of directed areas for faces from given region
[[nodiscard]] MRMESH_API Vector3d dirArea( const MeshTopology & topology, const VertCoords & points, const FaceBitSet & fs );

/// computes the sum of directed areas for faces from given region (or whole mesh)
[[nodiscard]] inline Vector3d dirArea( const MeshTopology & topology, const VertCoords & points, const FaceBitSet * fs = nullptr )
{
    return dirArea( topology, points, topology.getFaceIds( fs ) );
}

/// computes the sum of absolute projected area of faces from given region as visible if look from given direction
[[nodiscard]] MRMESH_API double projArea( const MeshTopology & topology, const VertCoords & points, const Vector3f & dir, const FaceBitSet & fs );

/// computes the sum of absolute projected area of faces from given region (or whole mesh) as visible if look from given direction
[[nodiscard]] inline double projArea( const MeshTopology & topology, const VertCoords & points, const Vector3f & dir, const FaceBitSet * fs = nullptr )
{
    return projArea( topology, points, dir, topology.getFaceIds( fs ) );
}

/// returns volume of the object surrounded by given region (or whole mesh if (region) is nullptr);
/// if the region has holes then each hole will be virtually filled by adding triangles for each edge and the hole's geometrical center
[[nodiscard]] MRMESH_API double volume( const MeshTopology & topology, const VertCoords & points, const FaceBitSet* region = nullptr );

/// computes the perimeter of the hole specified by one of its edges with no valid left face (left is hole)
[[nodiscard]] MRMESH_API double holePerimiter( const MeshTopology & topology, const VertCoords & points, EdgeId e );

/// computes directed area of the hole specified by one of its edges with no valid left face (left is hole);
/// if the hole is planar then returned vector is orthogonal to the plane pointing outside and its magnitude is equal to hole area
[[nodiscard]] MRMESH_API Vector3d holeDirArea( const MeshTopology & topology, const VertCoords & points, EdgeId e );

/// computes unit vector that is both orthogonal to given edge and to the normal of its left triangle, the vector is directed inside left triangle
[[nodiscard]] MRMESH_API Vector3f leftTangent( const MeshTopology & topology, const VertCoords & points, EdgeId e );

/// computes triangular face normal from its vertices
[[nodiscard]] inline Vector3f leftNormal( const MeshTopology & topology, const VertCoords & points, EdgeId e )
{
    return leftDirDblArea( topology, points, e ).normalized();
}

/// computes triangular face normal from its vertices
[[nodiscard]] inline Vector3f normal( const MeshTopology & topology, const VertCoords & points, FaceId f )
{
    return dirDblArea( topology, points, f ).normalized();
}

/// returns the plane containing given triangular face with normal looking outwards
[[nodiscard]] MRMESH_API Plane3f getPlane3f( const MeshTopology & topology, const VertCoords & points, FaceId f );
[[nodiscard]] MRMESH_API Plane3d getPlane3d( const MeshTopology & topology, const VertCoords & points, FaceId f );

/// computes sum of directed double areas of all triangles around given vertex
[[nodiscard]] MRMESH_API Vector3f dirDblArea( const MeshTopology & topology, const VertCoords & points, VertId v );

/// computes the length of summed directed double areas of all triangles around given vertex
[[nodiscard]] inline float dblArea( const MeshTopology & topology, const VertCoords & points, VertId v )
{
    return dirDblArea( topology, points, v ).length();
}

/// computes normal in a vertex using sum of directed areas of neighboring triangles
[[nodiscard]] inline Vector3f normal( const MeshTopology & topology, const VertCoords & points, VertId v )
{
    return dirDblArea( topology, points, v ).normalized();
}

/// computes normal in three vertices of p's triangle, then interpolates them using barycentric coordinates and normalizes again;
/// this is the same normal as in rendering with smooth shading
[[nodiscard]] MRMESH_API Vector3f normal( const MeshTopology & topology, const VertCoords & points, const MeshTriPoint & p );

/// computes angle-weighted sum of normals of incident faces of given vertex (only (region) faces will be considered);
/// the sum is normalized before returning
[[nodiscard]] MRMESH_API Vector3f pseudonormal( const MeshTopology & topology, const VertCoords & points, VertId v, const FaceBitSet * region = nullptr );

/// computes normalized half sum of face normals sharing given edge (only (region) faces will be considered);
[[nodiscard]] MRMESH_API Vector3f pseudonormal( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId e, const FaceBitSet * region = nullptr );

/// returns pseudonormal in corresponding face/edge/vertex for signed distance calculation
/// as suggested in the article "Signed Distance Computation Using the Angle Weighted Pseudonormal" by J. Andreas Baerentzen and Henrik Aanaes,
/// https://backend.orbit.dtu.dk/ws/portalfiles/portal/3977815/B_rentzen.pdf
/// unlike normal( const MeshTriPoint & p ), this is not a smooth function
[[nodiscard]] MRMESH_API Vector3f pseudonormal( const MeshTopology & topology, const VertCoords & points, const MeshTriPoint & p, const FaceBitSet * region = nullptr );

/// computes the sum of triangle angles at given vertex; optionally returns whether the vertex is on boundary
[[nodiscard]] MRMESH_API float sumAngles( const MeshTopology & topology, const VertCoords & points, VertId v, bool * outBoundaryVert = nullptr );

/// returns vertices where the sum of triangle angles is below given threshold
[[nodiscard]] MRMESH_API Expected<VertBitSet> findSpikeVertices( const MeshTopology & topology, const VertCoords & points, float minSumAngle, const VertBitSet* region = nullptr, const ProgressCallback& cb = {} );

/// given an edge between two triangular faces, computes sine of dihedral angle between them:
/// 0 if both faces are in the same plane,
/// positive if the faces form convex surface,
/// negative if the faces form concave surface
[[nodiscard]] MRMESH_API float dihedralAngleSin( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId e );

/// given an edge between two triangular faces, computes cosine of dihedral angle between them:
/// 1 if both faces are in the same plane,
/// 0 if the surface makes right angle turn at the edge,
/// -1 if the faces overlap one another
[[nodiscard]] MRMESH_API float dihedralAngleCos( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId e );

/// given an edge between two triangular faces, computes the dihedral angle between them:
/// 0 if both faces are in the same plane,
/// positive if the faces form convex surface,
/// negative if the faces form concave surface;
/// please consider the usage of faster dihedralAngleSin(e) and dihedralAngleCos(e)
[[nodiscard]] MRMESH_API float dihedralAngle( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId e );

/// computes discrete mean curvature in given vertex, measures in length^-1;
/// 0 for planar regions, positive for convex surface, negative for concave surface
[[nodiscard]] MRMESH_API float discreteMeanCurvature( const MeshTopology & topology, const VertCoords & points, VertId v );

/// computes discrete mean curvature in given edge, measures in length^-1;
/// 0 for planar regions, positive for convex surface, negative for concave surface
[[nodiscard]] MRMESH_API float discreteMeanCurvature( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId e );

/// computes discrete Gaussian curvature (or angle defect) at given vertex,
/// which 0 in inner vertices on planar mesh parts and reaches 2*pi on needle's tip, see http://math.uchicago.edu/~may/REU2015/REUPapers/Upadhyay.pdf
/// optionally returns whether the vertex is on boundary
[[nodiscard]] inline float discreteGaussianCurvature( const MeshTopology & topology, const VertCoords & points, VertId v, bool * outBoundaryVert = nullptr )
{
    return 2 * PI_F - sumAngles( topology, points, v, outBoundaryVert );
}

/// finds all mesh edges where dihedral angle is distinct from planar PI angle on at least given value
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet findCreaseEdges( const MeshTopology & topology, const VertCoords & points, float angleFromPlanar );

/// computes cotangent of the angle in the left( e ) triangle opposite to e,
/// and returns 0 if left face does not exist
[[nodiscard]] MRMESH_API float leftCotan( const MeshTopology & topology, const VertCoords & points, EdgeId e );

/// computes sum of cotangents of the angle in the left and right triangles opposite to given edge,
/// consider cotangents zero for not existing triangles
[[nodiscard]] inline float cotan( const MeshTopology & topology, const VertCoords & points, UndirectedEdgeId ue )
{
    EdgeId e{ ue };
    return leftCotan( topology, points, e ) + leftCotan( topology, points, e.sym() );
}

/// computes quadratic form in the vertex as the sum of squared distances from
/// 1) planes of adjacent triangles, with the weight equal to the angle of adjacent triangle at this vertex divided on PI in case of angleWeigted=true;
/// 2) lines of adjacent boundary and crease edges
[[nodiscard]] MRMESH_API QuadraticForm3f quadraticForm( const MeshTopology & topology, const VertCoords & points, VertId v, bool angleWeigted,
    const FaceBitSet * region = nullptr, const UndirectedEdgeBitSet * creases = nullptr );

/// passes through all given faces (or whole mesh if region == null) and finds the minimal bounding box containing all of them
/// if toWorld transformation is given then returns minimal bounding box in world space
[[nodiscard]] MRMESH_API Box3f computeBoundingBox( const MeshTopology & topology, const VertCoords & points, const FaceBitSet* region, const AffineXf3f* toWorld = nullptr );

/// computes average length of an edge in the mesh given by (topology, points)
[[nodiscard]] MRMESH_API float averageEdgeLength( const MeshTopology & topology, const VertCoords & points );

/// computes average position of all valid mesh vertices
[[nodiscard]] MRMESH_API Vector3f findCenterFromPoints( const MeshTopology & topology, const VertCoords & points );

/// computes center of mass considering that density of all triangles is the same
[[nodiscard]] MRMESH_API Vector3f findCenterFromFaces( const MeshTopology & topology, const VertCoords & points );

/// computes bounding box and returns its center
[[nodiscard]] MRMESH_API Vector3f findCenterFromBBox( const MeshTopology & topology, const VertCoords & points );

} //namespace MR
