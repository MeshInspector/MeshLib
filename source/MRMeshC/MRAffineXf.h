#pragma once

#include "MRMeshFwd.h"
#include "MRMatrix3.h"

MR_EXTERN_C_BEGIN

/// affine transformation: y = A*x + b, where A in VxV, and b in V
typedef struct MRAffineXf3f
{
    MRMatrix3f A;
    MRVector3f b;
} MRAffineXf3f;

/// initializes a default instance
MRMESHC_API MRAffineXf3f mrAffineXf3fNew( void );

/// creates translation-only transformation (with identity linear component)
MRMESHC_API MRAffineXf3f mrAffineXf3fTranslation( const MRVector3f* b );

/// creates linear-only transformation (without translation)
MRMESHC_API MRAffineXf3f mrAffineXf3fLinear( const MRMatrix3f* A );

/// composition of two transformations:
/// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
MRMESHC_API MRAffineXf3f mrAffineXf3fMul( const MRAffineXf3f* a, const MRAffineXf3f* b );

/// application of the transformation to a point
MRMESHC_API MRVector3f mrAffineXf3fApply( const MRAffineXf3f* xf, const MRVector3f* v );

MR_EXTERN_C_END
