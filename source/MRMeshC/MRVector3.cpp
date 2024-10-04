#include "MRVector3.h"

#include "MRMesh/MRVector3.h"

using namespace MR;

static_assert( sizeof( MRVector3f ) == sizeof( Vector3f ) );

MRVector3f mrVector3fDiagonal( float a )
{
    const auto res = Vector3f::diagonal( a );
    return *reinterpret_cast<const MRVector3f*>( &res );
}

MRVector3f mrVector3fPlusX()
{
    static const auto res = Vector3f::plusX();
    return *reinterpret_cast<const MRVector3f*>( &res );
}

MRVector3f mrVector3fPlusY()
{
    static const auto res = Vector3f::plusY();
    return *reinterpret_cast<const MRVector3f*>( &res );
}

MRVector3f mrVector3fPlusZ()
{
    static const auto res = Vector3f::plusZ();
    return *reinterpret_cast<const MRVector3f*>( &res );
}

MRVector3f mrVector3fAdd( const MRVector3f* a_, const MRVector3f* b_ )
{
    const auto& a = *reinterpret_cast<const Vector3f*>( a_ );
    const auto& b = *reinterpret_cast<const Vector3f*>( b_ );

    const auto res = a + b;
    return *reinterpret_cast<const MRVector3f*>( &res );
}

MRVector3f mrVector3fMul( const MRVector3f* a_, float b )
{
    const auto& a = *reinterpret_cast<const Vector3f*>( a_ );

    const auto res = a * b;
    return *reinterpret_cast<const MRVector3f*>( &res );
}

float mrVector3fLengthSq( const MRVector3f* v_ )
{
    const auto& v = *reinterpret_cast<const Vector3f*>( v_ );

    return v.lengthSq();
}

float mrVector3fLength( const MRVector3f* v_ )
{
    const auto& v = *reinterpret_cast<const Vector3f*>( v_ );

    return v.length();
}
