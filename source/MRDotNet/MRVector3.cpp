#include "MRVector3.h"
#pragma managed( push, off )
#include <MRMesh/MRVector3.h>
#pragma managed( pop )

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

Vector3f^ Vector3f::Diagonal( float a )
{
    return gcnew Vector3f( a, a, a );
}

inline Vector3f^ Vector3f::PlusX()
{
    return gcnew Vector3f( 1, 0, 0 );
}

inline Vector3f^ Vector3f::MinusX()
{
    return gcnew Vector3f( -1, 0, 0 );
}

inline Vector3f^ Vector3f::PlusY()
{
    return gcnew Vector3f( 0, 1, 0 );
}

inline Vector3f^ Vector3f::MinusY()
{
    return gcnew Vector3f( 0, -1, 0 );
}

inline Vector3f^ Vector3f::PlusZ()
{
    return gcnew Vector3f( 0, 0, 1 );
}

inline Vector3f^ Vector3f::MinusZ()
{
    return gcnew Vector3f( 0, 0, -1 );
}

Vector3f^ Vector3f::operator * ( float a, Vector3f^ b )
{
    if ( !b )
        throw gcnew System::ArgumentNullException( "b" );

    return gcnew Vector3f( new MR::Vector3f( a * *b->vec_ ) );
}

Vector3f^ Vector3f::operator * ( Vector3f^ a, float b )
{
    if ( !a )
        throw gcnew System::ArgumentNullException( "a" );

    return gcnew Vector3f( new MR::Vector3f( *a->vec_ * b ) );
}

Vector3f^ Vector3f::operator + ( Vector3f^ a, Vector3f^ b )
{
    if ( !a )
        throw gcnew System::ArgumentNullException( "a" );
    if ( !b )
        throw gcnew System::ArgumentNullException( "b" );

    return gcnew Vector3f( new MR::Vector3f( *a->vec_ + *b->vec_ ) );
}

Vector3f^ Vector3f::operator - ( Vector3f^ a, Vector3f^ b )
{
    if ( !a )
        throw gcnew System::ArgumentNullException( "a" );
    if ( !b )
        throw gcnew System::ArgumentNullException( "b" );

    return gcnew Vector3f( new MR::Vector3f( *a->vec_ - *b->vec_ ) );
}

float Vector3f::X::get()
{
    return vec_->x;
}

void Vector3f::X::set( float value )
{
    vec_->x = value;
}

float Vector3f::Y::get()
{
    return vec_->y;
}

void Vector3f::Y::set( float value )
{
    vec_->y = value;
}

float Vector3f::Z::get()
{
    return vec_->z;
}

void Vector3f::Z::set( float value )
{
    vec_->z = value;
}

bool Vector3f::operator ==( Vector3f^ a, Vector3f^ b )
{
    if ( !a )
        throw gcnew System::ArgumentNullException( "a" );
    if ( !b )
        throw gcnew System::ArgumentNullException( "b" );

    return *a->vec_ == *b->vec_;
}

bool Vector3f::operator != ( Vector3f^ a, Vector3f^ b )
{
    if ( !a )
        throw gcnew System::ArgumentNullException( "a" );
    if ( !b )
        throw gcnew System::ArgumentNullException( "b" );

    return *a->vec_ != *b->vec_;
}

MR_DOTNET_NAMESPACE_END
