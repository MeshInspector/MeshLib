#pragma once

#include "MRMeshFwd.h"

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

MR_EXTERN_C_END
