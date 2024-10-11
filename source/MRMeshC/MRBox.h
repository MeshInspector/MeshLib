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
MRMESHC_API MRBox3f mrBox3fNew( void );

/// creates box with given min value and size
MRMESHC_API MRBox3f mrBox3fFromMinAndSize( const MRVector3f* min, const MRVector3f* size );

/// true if the box contains at least one point
MRMESHC_API bool mrBox3fValid( const MRBox3f* box );

/// computes size of the box in all dimensions
MRMESHC_API MRVector3f mrBox3fSize( const MRBox3f* box );

/// computes length from min to max
MRMESHC_API float mrBox3fDiagonal( const MRBox3f* box );

/// computes the volume of this box
MRMESHC_API float mrBox3fVolume( const MRBox3f* box );

/// computes center of the box
MRMESHC_API MRVector3f mrBox3fCenter( const MRBox3f* box );

MR_EXTERN_C_END
