#include "MRVector3.h"

MR_DOTNET_NAMESPACE_BEGIN

Vector3f::Vector3f()
{
    vec_ = new MR::Vector3f();
}

Vector3f::Vector3f( MR::Vector3f* vec )
{
    vec_ = vec;
}

Vector3f::Vector3f( float x, float y, float z )
{
    vec_ = new MR::Vector3f( x, y, z );
}

Vector3f::~Vector3f()
{
    delete vec_;
}

Vector3f^ Vector3f::diagonal( float a )
{
    return gcnew Vector3f( a, a, a );
}

inline Vector3f^ Vector3f::plusX()
{
    return gcnew Vector3f( 1, 0, 0 );
}

inline Vector3f^ Vector3f::minusX()
{
    return gcnew Vector3f( -1, 0, 0 );
}

inline Vector3f^ Vector3f::plusY()
{
    return gcnew Vector3f( 0, 1, 0 );
}

inline Vector3f^ Vector3f::minusY()
{
    return gcnew Vector3f( 0, -1, 0 );
}

inline Vector3f^ Vector3f::plusZ()
{
    return gcnew Vector3f( 0, 0, 1 );
}

inline Vector3f^ Vector3f::minusZ()
{
    return gcnew Vector3f( 0, 0, -1 );
}

Vector3f^ Vector3f::operator * ( float a, Vector3f^ b )
{
    return gcnew Vector3f( new MR::Vector3f( a * *b->vec_ ) );
}

Vector3f^ Vector3f::operator * ( Vector3f^ a, float b )
{
    return gcnew Vector3f( new MR::Vector3f( *a->vec_ * b ) );
}

Vector3f^ Vector3f::operator + ( Vector3f^ a, Vector3f^ b )
{
    return gcnew Vector3f( new MR::Vector3f( *a->vec_ + *b->vec_ ) );
}

Vector3f^ Vector3f::operator - ( Vector3f^ a, Vector3f^ b )
{
    return gcnew Vector3f( new MR::Vector3f( *a->vec_ - *b->vec_ ) );
}

MR_DOTNET_NAMESPACE_END
