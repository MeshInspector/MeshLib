#include "MRBox.h"

#include "MRMesh/MRBox.h"

using namespace MR;

static_assert( sizeof( MRBox3f ) == sizeof( Box3f ) );

MRBox3f mrBox3fNew( void )
{
    static const Box3f res;
    return reinterpret_cast<const MRBox3f&>( res );
}

bool mrBox3fValid( const MRBox3f* box_ )
{
    const auto& box = *reinterpret_cast<const Box3f*>( box_ );

    return box.valid();
}

MRVector3f mrBox3fSize( const MRBox3f* box_ )
{
    const auto& box = *reinterpret_cast<const Box3f*>( box_ );

    const auto res = box.size();
    return reinterpret_cast<const MRVector3f&>( res );
}

float mrBox3fDiagonal( const MRBox3f* box_ )
{
    const auto& box = *reinterpret_cast<const Box3f*>( box_ );

    return box.diagonal();
}

float mrBox3fVolume( const MRBox3f* box_ )
{
    const auto& box = *reinterpret_cast<const Box3f*>( box_ );

    return box.volume();
}

MRVector3f mrBox3fCenter( const MRBox3f* box_ )
{
    const auto& box = *reinterpret_cast<const Box3f*>( box_ );

    auto result = box.center();

    return reinterpret_cast<const MRVector3f&>( result );
}

MRBox3f mrBox3fFromMinAndSize( const MRVector3f* min_, const MRVector3f* size_ )
{
    const auto& min = *reinterpret_cast<const Vector3f*>( min_ );
    const auto& size = *reinterpret_cast<const Vector3f*>( size_ );

    auto result = Box3f::fromMinAndSize( min, size );

    return reinterpret_cast<const MRBox3f&>( result );
}
