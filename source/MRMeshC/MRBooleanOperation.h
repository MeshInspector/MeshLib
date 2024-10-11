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

/// Input object index enum
typedef enum MRBooleanResultMapperMapObject
{
    MRBooleanResultMapperMapObjectA,
    MRBooleanResultMapperMapObjectB,
    MRBooleanResultMapperMapObjectCount
} MRBooleanResultMapperMapObject;

/**
 * \struct MRBooleanResultMapper
 * \brief Structure to map old mesh BitSets to new
 * \details Structure to easily map topology of mrBoolean input meshes to result mesh
 *
 * This structure allows to map faces, vertices and edges of mesh `A` and mesh `B` input of mrBoolean to result mesh topology primitives
 * \sa \ref mrBoolean
 */
typedef struct MRBooleanResultMapper MRBooleanResultMapper;

typedef struct MRBooleanResultMapperMaps MRBooleanResultMapperMaps;

/// creates a new BooleanResultMapper object
MRMESHC_API MRBooleanResultMapper* mrBooleanResultMapperNew( void );

/// Returns faces bitset of result mesh corresponding input one
MRMESHC_API MRFaceBitSet* mrBooleanResultMapperMapFaces( const MRBooleanResultMapper* mapper, const MRFaceBitSet* oldBS, MRBooleanResultMapperMapObject obj );

/// Returns vertices bitset of result mesh corresponding input one
MRMESHC_API MRVertBitSet* mrBooleanResultMapperMapVerts( const MRBooleanResultMapper* mapper, const MRVertBitSet* oldBS, MRBooleanResultMapperMapObject obj );

/// Returns edges bitset of result mesh corresponding input one
MRMESHC_API MREdgeBitSet* mrBooleanResultMapperMapEdges( const MRBooleanResultMapper* mapper, const MREdgeBitSet* oldBS, MRBooleanResultMapperMapObject obj );

/// Returns only new faces that are created during boolean operation
MRMESHC_API MRFaceBitSet* mrBooleanResultMapperNewFaces( const MRBooleanResultMapper* mapper );

/// returns updated oldBS leaving only faces that has corresponding ones in result mesh
MRMESHC_API MRFaceBitSet* mrBooleanResultMapperFilteredOldFaceBitSet( MRBooleanResultMapper* mapper, const MRFaceBitSet* oldBS, MRBooleanResultMapperMapObject obj );

MRMESHC_API const MRBooleanResultMapperMaps* mrBooleanResultMapperGetMaps( const MRBooleanResultMapper* mapper, MRBooleanResultMapperMapObject index );

/// "after cut" faces to "origin" faces
/// this map is not 1-1, but N-1
MRMESHC_API const MRFaceMap mrBooleanResultMapperMapsCut2origin( const MRBooleanResultMapperMaps* maps );

/// "after cut" faces to "after stitch" faces (1-1)
MRMESHC_API const MRFaceMap mrBooleanResultMapperMapsCut2newFaces( const MRBooleanResultMapperMaps* maps );

/// "origin" edges to "after stitch" edges (1-1)
MRMESHC_API const MRWholeEdgeMap mrBooleanResultMapperMapsOld2newEdges( const MRBooleanResultMapperMaps* maps );

/// "origin" vertices to "after stitch" vertices (1-1)
MRMESHC_API const MRVertMap mrBooleanResultMapperMapsOld2NewVerts( const MRBooleanResultMapperMaps* maps );

/// old topology indexes are valid if true
MRMESHC_API bool mrBooleanResultMapperMapsIdentity( const MRBooleanResultMapperMaps* maps );

/// deallocates a BooleanResultMapper object
MRMESHC_API void mrBooleanResultMapperFree( MRBooleanResultMapper* mapper );

MR_EXTERN_C_END
