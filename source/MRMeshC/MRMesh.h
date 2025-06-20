#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRBox.h"
#include "MRId.h"
#include "MRMeshTopology.h"
#include "MRMeshTriPoint.h"

MR_EXTERN_C_BEGIN

/// creates a copy of a Mesh object
MRMESHC_API MRMesh* mrMeshCopy( const MRMesh* mesh );

/// constructs a mesh from vertex coordinates and a set of triangles with given ids
MRMESHC_API MRMesh* mrMeshFromTriangles( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

/// constructs a mesh from vertex coordinates and a set of triangles with given ids;
/// unlike simple \ref mrMeshFromTriangles it tries to resolve non-manifold vertices by creating duplicate vertices
MRMESHC_API MRMesh* mrMeshFromTrianglesDuplicatingNonManifoldVertices( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

/// constructs a mesh from point triples;
/// \param duplicateNonManifoldVertices = false, all coinciding points are given the same VertId in the result;
/// \param duplicateNonManifoldVertices = true, it tries to avoid non-manifold vertices by creating duplicate vertices with same coordinates
MRMESHC_API MRMesh* mrMeshNewFromPointTriples( const MRTriangle3f* posTriangles, size_t posTrianglesNum, bool duplicateNonManifoldVertices );

/// gets read-only access to the mesh vertex coordinates
MRMESHC_API const MRVector3f* mrMeshPoints( const MRMesh* mesh );

/// gets read-write access to the mesh vertex coordinates
MRMESHC_API MRVector3f* mrMeshPointsRef( MRMesh* mesh );

/// gets count of the mesh vertex coordinates
MRMESHC_API size_t mrMeshPointsNum( const MRMesh* mesh );

/// gets read-only access to the mesh topology object
MRMESHC_API const MRMeshTopology* mrMeshTopology( const MRMesh* mesh );

/// gets read-write access to the mesh topology object
MRMESHC_API MRMeshTopology* mrMeshTopologyRef( MRMesh* mesh );

/// passes through all valid vertices and finds the minimal bounding box containing all of them;
/// if toWorld transformation is given then returns minimal bounding box in world space
MRMESHC_API MRBox3f mrMeshComputeBoundingBox( const MRMesh* mesh, const MRAffineXf3f* toWorld );

/// applies given transformation to specified vertices
/// if region is NULL, all valid mesh vertices are used
MRMESHC_API void mrMeshTransform( MRMesh* mesh, const MRAffineXf3f* xf, const MRVertBitSet* region );

/// computes directed area of the hole specified by one of its edges with no valid left face (left is hole);
/// if the hole is planar then returned vector is orthogonal to the plane pointing outside and its magnitude is equal to hole area
MRMESHC_API MRVector3f mrMeshHoleDirArea( const MRMesh* mesh, MREdgeId e );

/// computes the area of given face-region (or whole mesh if region is null)
MRMESHC_API double mrMeshArea( const MRMesh* mesh, const MRFaceBitSet* region );

/// returns Euclidean length of the edge
MRMESHC_API float mrMeshEdgeLength( const MRMesh* mesh, MRUndirectedEdgeId e );
/// returns squared Euclidean length of the edge (faster to compute than length)
MRMESHC_API float mrMeshEdgeLengthSq( const MRMesh* mesh, MRUndirectedEdgeId e );

/// deletes multiple given faces, also deletes adjacent edges and vertices if they were not shared by remaining faces and not in \param keepEdges
MRMESHC_API void mrMeshDeleteFaces( MRMesh* mesh, const MRFaceBitSet* fs, const MRUndirectedEdgeBitSet* keepEdges );

/// optional parameters for \ref mrMeshAddMeshPart
typedef struct MRMeshAddMeshPartParameters
{
    /// if flipOrientation then every from triangle is inverted before adding
    bool flipOrientation;
    /// contours on this mesh that have to be stitched with
    const MREdgePath* thisContours;
    size_t thisContoursNum;
    /// contours on from mesh during addition
    const MREdgePath* fromContours;
    size_t fromContoursNum;
    // TODO: map
} MRMeshAddMeshPartParameters;

/// appends mesh (from) in addition to this mesh: creates new edges, faces, verts and points
MRMESHC_API void mrMeshAddMeshPart( MRMesh* mesh, const MRMeshPart* meshPart, const MRMeshAddMeshPartParameters* params );

/// tightly packs all arrays eliminating lone edges and invalid face, verts and points
MRMESHC_API void mrMeshPack( MRMesh* mesh, bool rearrangeTriangles );

/// packs tightly and rearranges vertices, triangles and edges to put close in space elements in close indices
/// \param preserveAABBTree whether to keep valid mesh's AABB tree after return (it will take longer to compute and it will occupy more memory)
MRMESHC_API void mrMeshPackOptimally( MRMesh* mesh, bool preserveAABBTree );

/// returns volume of closed mesh region, if region is not closed DBL_MAX is returned
/// if region is NULL - whole mesh is region
MRMESHC_API double mrMeshVolume( const MRMesh* mesh, const MRFaceBitSet* region );

/// deallocates a Mesh object
MRMESHC_API void mrMeshFree( MRMesh* mesh );

/// returns three vertex ids for valid triangles (which can be accessed by FaceId),
/// vertex ids for invalid triangles are undefined, and shall not be read
/// NOTE: this is a shortcut for mrMeshTopologyGetTriangulation( mrMeshTopology( mesh ) )
MRMESHC_API MRTriangulation* mrMeshGetTriangulation( const MRMesh* mesh );

/// returns one edge with no valid left face for every boundary in the mesh
/// NOTE: this is a shortcut for mrMeshTopologyFindHoleRepresentiveEdges( mrMeshTopology( mesh ) )
MRMESHC_API MREdgePath* mrMeshFindHoleRepresentiveEdges( const MRMesh* mesh );

/// invalidates caches (aabb-trees) after any change in mesh geometry or topology
/// \param pointsChanged specifies whether points have changed (otherwise only topology has changed)
MRMESHC_API void mrMeshInvalidateCaches( MRMesh* mesh, bool pointsChanged );

/// appends another mesh as separate connected component(s) to this
// TODO: outFmap, outVmap, outEmap, rearrangeTriangles
MRMESHC_API void mrMeshAddMesh( MRMesh* mesh, const MRMesh* from );

/// computes normal in a vertex using sum of directed areas of neighboring triangles
MRMESHC_API MRVector3f mrMeshNormalFromVert( const MRMesh* mesh, MRVertId v );

/// converts face id and 3d point into barycentric representation
MRMESHC_API MRMeshTriPoint mrToTriPoint( const MRMesh* mesh, MRFaceId f, MRVector3f point );

MR_EXTERN_C_END
