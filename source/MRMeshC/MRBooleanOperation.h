#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

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

#ifdef __cplusplus
}
#endif
