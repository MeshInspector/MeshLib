#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

MRMESHC_API MRMesh* mrMeshCopy( const MRMesh* mesh );

MRMESHC_API MRMesh* mrMeshFromTriangles( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const int* t, size_t tNum );

MRMESHC_API MRMesh* mrMeshFromTrianglesDuplicatingNonManifoldVertices( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const int* t, size_t tNum );

MRMESHC_API MRMesh* mrMeshNewFromPointTriples( const MRVector3f* posTriangles, size_t posTrianglesNum, bool duplicateNonManifoldVertices );

MRMESHC_API const MRVector3f* mrMeshPoints( const MRMesh* mesh );

MRMESHC_API size_t mrMeshPointsNum( const MRMesh* mesh );

MRMESHC_API const MRMeshTopology* mrMeshTopology( const MRMesh* mesh );

MRMESHC_API MRMeshTopology* mrMeshTopologyRef( MRMesh* mesh );

MRMESHC_API void mrMeshFree( MRMesh* mesh );

MR_EXTERN_C_END
