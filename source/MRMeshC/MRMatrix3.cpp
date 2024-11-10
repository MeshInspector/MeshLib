#include "MRMatrix3.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMatrix3.h"

using namespace MR;

REGISTER_AUTO_CAST( Matrix3f )
REGISTER_AUTO_CAST( Vector3f )

static_assert( sizeof( MRMatrix3f ) == sizeof( Matrix3f ) );

MRMatrix3f mrMatrix3fIdentity()
{
    static const auto result = Matrix3f::identity();
    RETURN( result );
}

MRMatrix3f mrMatrix3fRotationVector( const MRVector3f* from_, const MRVector3f* to_ )
{
    ARG( from ); ARG( to );
    RETURN( Matrix3f::rotation( from, to ) );
}

MRMatrix3f mrMatrix3fRotationScalar( const MRVector3f* axis_, float angle )
{
    ARG( axis );
    RETURN( Matrix3f::rotation( axis, angle ) );
}

MRMatrix3f mrMatrix3fMul( const MRMatrix3f* a_, const MRMatrix3f* b_ )
{
    ARG( a ); ARG( b );
    RETURN( a * b );
}

MRMatrix3f mrMatrix3fAdd( const MRMatrix3f* a_, const MRMatrix3f* b_ )
{
    ARG( a ); ARG( b );
    RETURN( a + b );
}

MRMatrix3f mrMatrix3fSub( const MRMatrix3f* a_, const MRMatrix3f* b_ )
{
    ARG( a ); ARG( b );
    RETURN( a - b );
}

MRVector3f mrMatrix3fMulVector( const MRMatrix3f* a_, const MRVector3f* b_ )
{
    ARG( a ); ARG( b );
    RETURN( a * b );
}

bool mrMatrix3fEqual( const MRMatrix3f* a_, const MRMatrix3f* b_ )
{
    ARG( a ); ARG( b );
    RETURN( a == b );
}


