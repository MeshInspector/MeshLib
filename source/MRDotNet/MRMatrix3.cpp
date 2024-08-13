#include "MRMatrix3.h"
#include "MRVector3.h"

#pragma managed( push, off )
#include <MRMesh/MRMatrix3.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

Matrix3f::Matrix3f()
{
    mat_ = new MR::Matrix3f();
    X = Vector3f::PlusX();
    Y = Vector3f::PlusY();
    Z = Vector3f::PlusZ();    
}

Matrix3f::Matrix3f( Vector3f^ x, Vector3f^ y, Vector3f^ z )
{
    if ( !x || !y || !z )
        throw gcnew System::ArgumentNullException();

    mat_ = new MR::Matrix3f();
    this->X = x;
    this->Y = y;
    this->Z = z;    
}

Matrix3f::~Matrix3f()
{
    delete mat_;
}

Matrix3f::Matrix3f( MR::Matrix3f* mat )
{
    mat_ = mat;
    X = gcnew Vector3f( new MR::Vector3f( mat_->x ) );
    Y = gcnew Vector3f( new MR::Vector3f( mat_->y ) );
    Z = gcnew Vector3f( new MR::Vector3f( mat_->z ) );
}

Matrix3f^ Matrix3f::Zero()
{
    return gcnew Matrix3f( gcnew Vector3f(), gcnew Vector3f(), gcnew Vector3f() );
}

Matrix3f^ Matrix3f::Rotation( Vector3f^ axis, float angle )
{
    if ( !axis )
        throw gcnew System::ArgumentNullException( "axis" );

    return gcnew Matrix3f( new MR::Matrix3f( MR::Matrix3f::rotation( *axis->vec(), angle) ) );
}

Matrix3f^ Matrix3f::Rotation( Vector3f^ from, Vector3f^ to )
{
    if ( !from || !to )
        throw gcnew System::ArgumentNullException();

    return gcnew Matrix3f( new MR::Matrix3f( MR::Matrix3f::rotation( *from->vec(), *to->vec() ) ) );
}

Matrix3f^ Matrix3f::operator*( Matrix3f^ a, Matrix3f^ b )
{
    if ( !a || !b )
        throw gcnew System::ArgumentNullException();

    return gcnew Matrix3f( new MR::Matrix3f( *a->mat_ * *b->mat_ ) );
}

Vector3f^ Matrix3f::operator*( Matrix3f^ a, Vector3f^ b )
{
    if ( !a )
        throw gcnew System::ArgumentNullException();

    return gcnew Vector3f( new MR::Vector3f( *a->mat_ * *b->vec() ) );
}

Vector3f^ Matrix3f::X::get()
{
    return x_;
}

void Matrix3f::X::set( Vector3f^ value )
{
    if ( !value )
        throw gcnew System::ArgumentNullException();

    x_ = value;
    mat_->x = *value->vec();
}

Vector3f^ Matrix3f::Y::get()
{
    return y_;
}

void Matrix3f::Y::set( Vector3f^ value )
{
    if ( !value )
        throw gcnew System::ArgumentNullException();
    y_ = value;
    mat_->y = *value->vec();
}

Vector3f^ Matrix3f::Z::get()
{
    return z_;
}

void Matrix3f::Z::set( Vector3f^ value )
{
    if ( !value )
        throw gcnew System::ArgumentNullException();

    z_ = value;
    mat_->z = *value->vec();
}

bool Matrix3f::operator==( Matrix3f^ a, Matrix3f^ b )
{
    if ( !a || !b )
        throw gcnew System::ArgumentNullException();

    return *a->mat_ == *b->mat_;
}

bool Matrix3f::operator!=( Matrix3f^ a, Matrix3f^ b )
{
    if ( !a || !b )
        throw gcnew System::ArgumentNullException();

    return *a->mat_ != *b->mat_;
}

Matrix3f^ Matrix3f::operator-( Matrix3f^ a, Matrix3f^ b )
{
    return gcnew Matrix3f( new MR::Matrix3f( *a->mat_ - *b->mat_ ) );
}

Matrix3f^ Matrix3f::operator+( Matrix3f^ a, Matrix3f^ b )
{
    return gcnew Matrix3f( new MR::Matrix3f( *a->mat_ + *b->mat_ ) );
}

MR_DOTNET_NAMESPACE_END
