#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRId.h"

MR_EXTERN_C_BEGIN

MRMESHC_API MRMesh* mrMeshCopy( const MRMesh* mesh );

MRMESHC_API MRMesh* mrMeshFromTriangles( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

MRMESHC_API MRMesh* mrMeshFromTrianglesDuplicatingNonManifoldVertices( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

MRMESHC_API MRMesh* mrMeshNewFromPointTriples( const MRTriangle3f* posTriangles, size_t posTrianglesNum, bool duplicateNonManifoldVertices );

MRMESHC_API const MRVector3f* mrMeshPoints( const MRMesh* mesh );

MRMESHC_API MRVector3f* mrMeshPointsRef( MRMesh* mesh );

MRMESHC_API size_t mrMeshPointsNum( const MRMesh* mesh );

MRMESHC_API const MRMeshTopology* mrMeshTopology( const MRMesh* mesh );

MRMESHC_API MRMeshTopology* mrMeshTopologyRef( MRMesh* mesh );

MRMESHC_API void mrMeshTransform( MRMesh* mesh, const MRAffineXf3f* xf, const MRVertBitSet* region );

typedef struct MRMESHC_CLASS MRMeshAddPartByMaskParameters
{
    bool flipOrientation;
    const MREdgePath* thisContours;
    size_t thisContoursNum;
    const MREdgePath* fromContours;
    size_t fromContoursNum;
    // TODO: map
} MRMeshAddPartByMaskParameters;

MRMESHC_API void mrMeshAddPartByMask( MRMesh* mesh, const MRMesh* from, const MRFaceBitSet* fromFaces, const MRMeshAddPartByMaskParameters* params );

MRMESHC_API void mrMeshFree( MRMesh* mesh );

MR_EXTERN_C_END
