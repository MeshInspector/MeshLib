#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

MR_EXTERN_C_BEGIN

typedef struct MRBox3f
{
    MRVector3f min;
    MRVector3f max;
} MRBox3f;

/// creates invalid box by default
MRMESHC_API MRBox3f mrBox3fNew();

/// true if the box contains at least one point
MRMESHC_API bool mrBox3fValid( const MRBox3f* box );

/// computes size of the box in all dimensions
MRMESHC_API MRVector3f mrBox3fSize( const MRBox3f* box );

/// computes length from min to max
MRMESHC_API float mrBox3fDiagonal( const MRBox3f* box );

MR_EXTERN_C_END
