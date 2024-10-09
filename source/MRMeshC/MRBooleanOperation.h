#pragma once

#include "MRMeshFwd.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

/// Available CSG operations
typedef enum MRBooleanOperation
{
    /// Part of mesh `A` that is inside of mesh `B`
    MRBooleanOperationInsideA = 0,
    /// Part of mesh `B` that is inside of mesh `A`
    MRBooleanOperationInsideB,
    /// Part of mesh `A` that is outside of mesh `B`
    MRBooleanOperationOutsideA,
    /// Part of mesh `B` that is outside of mesh `A`
    MRBooleanOperationOutsideB,
    /// Union surface of two meshes (outside parts)
    MRBooleanOperationUnion,
    /// Intersection surface of two meshes (inside parts)
    MRBooleanOperationIntersection,
    /// Surface of mesh `B` - surface of mesh `A` (outside `B` - inside `A`)
    MRBooleanOperationDifferenceBA,
    /// Surface of mesh `A` - surface of mesh `B` (outside `A` - inside `B`)
    MRBooleanOperationDifferenceAB,
    /// not a valid operation
    MRBooleanOperationCount
} MRBooleanOperation;

/// ...
typedef enum MRBooleanResultMapperMapObject
{
    MRBooleanResultMapperMapObjectA,
    MRBooleanResultMapperMapObjectB,
    MRBooleanResultMapperMapObjectCount
} MRBooleanResultMapperMapObject;

/// ...
typedef struct MRBooleanResultMapper MRBooleanResultMapper;

/// ...
typedef struct MRBooleanResultMapperMaps MRBooleanResultMapperMaps;

/// ...
MRMESHC_API MRBooleanResultMapper* mrBooleanResultMapperNew( void );

/// ...
MRMESHC_API MRFaceBitSet* mrBooleanResultMapperMapFaces( const MRBooleanResultMapper* mapper, const MRFaceBitSet* oldBS, MRBooleanResultMapperMapObject obj );

/// ...
MRMESHC_API MRVertBitSet* mrBooleanResultMapperMapVerts( const MRBooleanResultMapper* mapper, const MRVertBitSet* oldBS, MRBooleanResultMapperMapObject obj );

/// ...
MRMESHC_API MREdgeBitSet* mrBooleanResultMapperMapEdges( const MRBooleanResultMapper* mapper, const MREdgeBitSet* oldBS, MRBooleanResultMapperMapObject obj );

/// ...
MRMESHC_API MRFaceBitSet* mrBooleanResultMapperNewFaces( const MRBooleanResultMapper* mapper );

/// ...
MRMESHC_API MRFaceBitSet* mrBooleanResultMapperFilteredOldFaceBitSet( MRBooleanResultMapper* mapper, const MRFaceBitSet* oldBS, MRBooleanResultMapperMapObject obj );

/// ...
MRMESHC_API const MRBooleanResultMapperMaps* mrBooleanResultMapperGetMaps( const MRBooleanResultMapper* mapper, MRBooleanResultMapperMapObject index );

/// ...
MRMESHC_API const MRFaceMap mrBooleanResultMapperMapsCut2origin( const MRBooleanResultMapperMaps* maps );

/// ...
MRMESHC_API const MRFaceMap mrBooleanResultMapperMapsCut2newFaces( const MRBooleanResultMapperMaps* maps );

/// ...
MRMESHC_API const MRWholeEdgeMap mrBooleanResultMapperMapsOld2newEdges( const MRBooleanResultMapperMaps* maps );

/// ...
MRMESHC_API const MRVertMap mrBooleanResultMapperMapsOld2NewVerts( const MRBooleanResultMapperMaps* maps );

/// ...
MRMESHC_API bool mrBooleanResultMapperMapsIdentity( const MRBooleanResultMapperMaps* maps );

MR_EXTERN_C_END
