#include "MRVector3.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRVector3.h"

using namespace MR;

REGISTER_AUTO_CAST( Vector3f )

static_assert( sizeof( MRVector3f ) == sizeof( Vector3f ) );

MRVector3f mrVector3fDiagonal( float a )
{
    RETURN( Vector3f::diagonal( a ) );
}

MRVector3f mrVector3fPlusX()
{
    static const auto res = Vector3f::plusX();
    RETURN( res );
}

MRVector3f mrVector3fPlusY()
{
    static const auto res = Vector3f::plusY();
    RETURN( res );
}

MRVector3f mrVector3fPlusZ()
{
    static const auto res = Vector3f::plusZ();
    RETURN( res );
}

MRVector3f mrVector3fAdd( const MRVector3f* a_, const MRVector3f* b_ )
{
    ARG( a ); ARG( b );
    RETURN( a + b );
}

MRVector3f mrVector3fSub( const MRVector3f* a_, const MRVector3f* b_ )
{
    ARG( a ); ARG( b );
    RETURN( a - b );
}

MRVector3f mrVector3fMulScalar( const MRVector3f* a_, float b )
{
    ARG( a );
    RETURN( a * b );
}

float mrVector3fLengthSq( const MRVector3f* v_ )
{
    ARG( v );
    return v.lengthSq();
}

float mrVector3fLength( const MRVector3f* v_ )
{
    ARG( v );
    return v.length();
}
