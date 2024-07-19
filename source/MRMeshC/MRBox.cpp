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
