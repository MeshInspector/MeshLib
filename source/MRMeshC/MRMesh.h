#pragma once

#include "MRMeshFwd.h"

#ifdef __cplusplus
extern "C"
{
#endif

MRMesh* mrMeshCopy( const MRMesh* mesh );

MRMesh* mrMeshFromTriangles( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const int* t, size_t tNum );

MRMesh* mrMeshFromTrianglesDuplicatingNonManifoldVertices( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const int* t, size_t tNum );

MRMesh* mrMeshNewFromPointTriples( const MRVector3f* posTriangles, size_t posTrianglesNum, bool duplicateNonManifoldVertices );

const MRVector3f* mrMeshPoints( const MRMesh* mesh );

size_t mrMeshPointsNum( const MRMesh* mesh );

const MRMeshTopology* mrMeshTopology( const MRMesh* mesh );

MRMeshTopology* mrMeshTopologyRef( MRMesh* mesh );

void mrMeshFree( MRMesh* mesh );

#ifdef __cplusplus
}
#endif
