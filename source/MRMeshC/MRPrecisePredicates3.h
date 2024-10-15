#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// float-to-int coordinate converter
typedef struct MRConvertToIntVector MRConvertToIntVector;

/// deallocates the ConvertToIntVector object
MRMESHC_API void mrConvertToIntVectorFree( MRConvertToIntVector* conv );

/// int-to-float coordinate converter
typedef struct MRConvertToFloatVector MRConvertToFloatVector;

/// deallocates the ConvertToFloatVector object
MRMESHC_API void mrConvertToFloatVectorFree( MRConvertToFloatVector* conv );

/// this struct contains coordinate converters float-int-float
typedef struct MRCoordinateConverters
{
    MRConvertToIntVector* toInt;
    MRConvertToFloatVector* toFloat;
} MRCoordinateConverters;

MR_EXTERN_C_END
