#include "MRMatrix3.h"

#include "MRMesh/MRMatrix3.h"

using namespace MR;

static_assert( sizeof( MRMatrix3f ) == sizeof( Matrix3f ) );

MRMatrix3f mrMatrix3fIdentity()
{
    constexpr auto m = Matrix3f::identity();
    return *reinterpret_cast<const MRMatrix3f*>( &m );
}
