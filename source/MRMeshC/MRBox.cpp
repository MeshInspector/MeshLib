#include "MRBox.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRBox.h"

using namespace MR;

REGISTER_AUTO_CAST( Box3f )
REGISTER_AUTO_CAST( Vector3f )

static_assert( sizeof( MRBox3f ) == sizeof( Box3f ) );

MRBox3f mrBox3fNew( void )
{
    static const Box3f result;
    RETURN( result );
}

bool mrBox3fValid( const MRBox3f* box_ )
{
    ARG( box );
    return box.valid();
}

MRVector3f mrBox3fSize( const MRBox3f* box_ )
{
    ARG( box );
    RETURN( box.size() );
}

float mrBox3fDiagonal( const MRBox3f* box_ )
{
    ARG( box );
    return box.diagonal();
}

float mrBox3fVolume( const MRBox3f* box_ )
{
    ARG( box );
    return box.volume();
}

MRVector3f mrBox3fCenter( const MRBox3f* box_ )
{
    ARG( box );
    RETURN( box.center() );
}

MRBox3f mrBox3fFromMinAndSize( const MRVector3f* min_, const MRVector3f* size_ )
{
    ARG( min ); ARG( size );
    RETURN( Box3f::fromMinAndSize( min, size ) );
}
