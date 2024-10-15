#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"

MR_EXTERN_C_BEGIN

/// \brief encodes a point inside a triangle using barycentric coordinates
/// \details Notations used below: v0, v1, v2 - points of the triangle
typedef struct MRTriPointf
{
    /// barycentric coordinates:
    /// a+b in [0,1], a+b=0 => point is in v0, a+b=1 => point is on [v1,v2] edge
    /// a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
    float a;
    /// b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2
    float b;
} MRTriPointf;

/// given a point coordinates and triangle (v0,v1,v2) computes barycentric coordinates of the point
MRMESHC_API MRTriPointf mrTriPointfFromTriangle( const MRVector3f* p, const MRVector3f* v0, const MRVector3f* v1, const MRVector3f* v2 );

/// given three values in three vertices, computes interpolated value at this barycentric coordinates
MRMESHC_API MRVector3f mrTriPointfInterpolate( const MRTriPointf* tp, const MRVector3f* v0, const MRVector3f* v1, const MRVector3f* v2 );

MR_EXTERN_C_END
