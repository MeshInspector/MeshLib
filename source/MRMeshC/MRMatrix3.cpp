#include "MRMatrix3.h"

#include "MRMesh/MRMatrix3.h"

#include <cstring>

using namespace MR;

static_assert( sizeof( MRMatrix3f ) == sizeof( Matrix3f ) );

MRMatrix3f mrMatrix3fIdentity()
{
    constexpr auto m = Matrix3f::identity();
    MRMatrix3f res;
    std::memcpy( &res, &m, sizeof( m ) );
    return res;
}
