#include "MRTriPoint.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRTriPoint.h"

using namespace MR;

static_assert( sizeof( MRTriPointf ) == sizeof( MR::TriPointf ) );

REGISTER_AUTO_CAST( TriPointf )
REGISTER_AUTO_CAST( Vector3f )

MRTriPointf mrTriPointfFromTriangle( const MRVector3f* p_, const MRVector3f* v0_, const MRVector3f* v1_, const MRVector3f* v2_ )
{
    ARG( p ); ARG( v0 ); ARG( v1 ); ARG( v2 );
    RETURN( TriPointf( p, v0, v1, v2 ) );
}

MRVector3f mrTriPointfInterpolate( const MRTriPointf* tp_, const MRVector3f* v0_, const MRVector3f* v1_, const MRVector3f* v2_ )
{
    ARG( tp ); ARG( v0 ); ARG( v1 ); ARG( v2 );
    RETURN( tp.interpolate( v0, v1, v2 ) );
}
