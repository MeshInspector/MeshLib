#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRId.h"

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

/// applies given transformation to specified vertices
/// if region is NULL, all valid mesh vertices are used
MRMESHC_API void mrMeshTransform( MRMesh* mesh, const MRAffineXf3f* xf, const MRVertBitSet* region );

/// optional parameters for \ref mrMeshAddPartByMask
typedef struct MRMeshAddPartByMaskParameters
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
} MRMeshAddPartByMaskParameters;

/// appends mesh (from) in addition to this mesh: creates new edges, faces, verts and points
MRMESHC_API void mrMeshAddPartByMask( MRMesh* mesh, const MRMesh* from, const MRFaceBitSet* fromFaces, const MRMeshAddPartByMaskParameters* params );

/// deallocates a Mesh object
MRMESHC_API void mrMeshFree( MRMesh* mesh );

MR_EXTERN_C_END
