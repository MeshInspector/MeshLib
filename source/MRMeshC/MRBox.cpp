#include "MRBox.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRBox.h"

using namespace MR;

REGISTER_AUTO_CAST( Box3f )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST( Box3i )
REGISTER_AUTO_CAST( Vector3i )

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

static_assert( sizeof( MRBox3i ) == sizeof( Box3i ) );

MRBox3i mrBox3iNew( void )
{
    static const Box3i result;
    RETURN( result );
}

bool mrBox3iValid( const MRBox3i* box_ )
{
    ARG( box );
    return box.valid();
}

MRVector3i mrBox3iSize( const MRBox3i* box_ )
{
    ARG( box );
    RETURN( box.size() );
}

int mrBox3iVolume( const MRBox3i* box_ )
{
    ARG( box );
    return box.volume();
}

MRVector3i mrBox3iCenter( const MRBox3i* box_ )
{
    ARG( box );
    RETURN( box.center() );
}

MRBox3i mrBox3iFromMinAndSize( const MRVector3i* min_, const MRVector3i* size_ )
{
    ARG( min ); ARG( size );
    RETURN( Box3i::fromMinAndSize( min, size ) );
}
