#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// ...
typedef struct MRConvertToIntVector MRConvertToIntVector;

/// ...
typedef struct MRConvertToFloatVector MRConvertToFloatVector;

/// ...
typedef struct MRCoordinateConverters
{
    MRConvertToIntVector* toInt;
    MRConvertToFloatVector* toFloat;
} MRCoordinateConverters;

MR_EXTERN_C_END
