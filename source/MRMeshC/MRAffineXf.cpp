#include "MRAffineXf.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRVector3.h"

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( Matrix3f )
REGISTER_AUTO_CAST( Vector3f )

static_assert( sizeof( MRAffineXf3f ) == sizeof( AffineXf3f ) );

MRAffineXf3f mrAffineXf3fNew()
{
    static const AffineXf3f result;
    RETURN( result );
}

MRAffineXf3f mrAffineXf3fTranslation( const MRVector3f* b_ )
{
    ARG( b );
    RETURN( AffineXf3f::translation( b ) );
}

MRAffineXf3f mrAffineXf3fLinear( const MRMatrix3f* A_ )
{
    ARG( A );
    RETURN( AffineXf3f::linear( A ) );
}

MRAffineXf3f mrAffineXf3fMul( const MRAffineXf3f* a_, const MRAffineXf3f* b_ )
{
    ARG( a ); ARG( b );
    RETURN( a * b );
}

MRVector3f mrAffineXf3fApply( const MRAffineXf3f* xf_, const MRVector3f* v_ )
{
    ARG( xf ); ARG( v );
    RETURN( xf( v ) );
}
