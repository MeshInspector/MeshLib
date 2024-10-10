#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// ...
typedef struct MRConvertToIntVector MRConvertToIntVector;

/// ...
MRMESHC_API void mrConvertToIntVectorFree( MRConvertToIntVector* conv );

/// ...
typedef struct MRConvertToFloatVector MRConvertToFloatVector;

/// ...
MRMESHC_API void mrConvertToFloatVectorFree( MRConvertToFloatVector* conv );

/// ...
typedef struct MRCoordinateConverters
{
    MRConvertToIntVector* toInt;
    MRConvertToFloatVector* toFloat;
} MRCoordinateConverters;

MR_EXTERN_C_END
