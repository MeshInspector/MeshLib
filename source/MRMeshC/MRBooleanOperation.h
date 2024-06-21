#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

enum MRBooleanOperation
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
};

MR_EXTERN_C_END
