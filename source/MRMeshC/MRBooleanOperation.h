#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

typedef enum MRBooleanOperation
{
    MRBooleanOperationInsideA = 0,
    MRBooleanOperationInsideB,
    MRBooleanOperationOutsideA,
    MRBooleanOperationOutsideB,
    MRBooleanOperationUnion,
    MRBooleanOperationIntersection,
    MRBooleanOperationDifferenceBA,
    MRBooleanOperationDifferenceAB,
    MRBooleanOperationCount
} MRBooleanOperation;

MR_EXTERN_C_END
