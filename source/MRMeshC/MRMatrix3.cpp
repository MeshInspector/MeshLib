#include "MRMatrix3.h"

#include "MRMesh/MRMatrix3.h"

#include <cstring>

using namespace MR;

static_assert( sizeof( MRMatrix3f ) == sizeof( Matrix3f ) );

MRMatrix3f mrMatrix3fIdentity()
{
    constexpr auto res = Matrix3f::identity();
    return *reinterpret_cast<const MRMatrix3f*>( &res );
}

MRMatrix3f mrMatrix3fRotationVector( const MRVector3f* from, const MRVector3f* to )
{
    const auto res = Matrix3f::rotation(
        *reinterpret_cast<const Vector3f*>( from ),
        *reinterpret_cast<const Vector3f*>( to )
    );
    return *reinterpret_cast<const MRMatrix3f*>( &res );
}

MRMatrix3f mrMatrix3fRotationScalar( const MRVector3f* axis_, float angle )
{
    const auto& axis = *reinterpret_cast<const Vector3f*>( axis_ );

    const auto res = Matrix3f::rotation( axis, angle );
    return *reinterpret_cast<const MRMatrix3f*>( &res );
}

MRMatrix3f mrMatrix3fMul( const MRMatrix3f* a_, const MRMatrix3f* b_ )
{
    const auto& a = *reinterpret_cast<const Matrix3f*>( a_ );
    const auto& b = *reinterpret_cast<const Matrix3f*>( b_ );

    const auto res = a * b;
    return *reinterpret_cast<const MRMatrix3f*>( &res );
}
