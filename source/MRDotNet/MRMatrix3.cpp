#include "MRMatrix3.h"
#include "MRVector3.h"

#pragma managed( push, off )
#include <MRMesh/MRMatrix3.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

Matrix3f::Matrix3f()
{
    x = Vector3f::plusX();
    y = Vector3f::plusY();
    z = Vector3f::plusZ();

    mat_ = new MR::Matrix3f( *x->vec(), *y->vec(), *z->vec() );
}

Matrix3f::Matrix3f( Vector3f^ x, Vector3f^ y, Vector3f^ z )
{
    this->x = x;
    this->y = y;
    this->z = z;

    mat_ = new MR::Matrix3f( *x->vec(), *y->vec(), *z->vec() );
}

Matrix3f::~Matrix3f()
{
    delete mat_;
}

Matrix3f::Matrix3f( MR::Matrix3f* mat )
{
    mat_ = mat;
}

Matrix3f^ Matrix3f::zero()
{
    return gcnew Matrix3f( gcnew Vector3f(), gcnew Vector3f(), gcnew Vector3f() );
}

Matrix3f^ Matrix3f::rotation( Vector3f^ axis, float angle )
{
    return gcnew Matrix3f( new MR::Matrix3f( MR::Matrix3f::rotation( *axis->vec(), angle) ) );
}

Matrix3f^ Matrix3f::rotation( Vector3f^ from, Vector3f^ to )
{
    return gcnew Matrix3f( new MR::Matrix3f( MR::Matrix3f::rotation( *from->vec(), *to->vec() ) ) );
}

Matrix3f^ Matrix3f::operator*( Matrix3f^ a, Matrix3f^ b )
{
    return gcnew Matrix3f( new MR::Matrix3f( *a->mat_ * *b->mat_ ) );
}

Vector3f^ Matrix3f::x::get()
{
    return gcnew Vector3f( new MR::Vector3f( mat_->x ) );
}

void Matrix3f::x::set( Vector3f^ value )
{
    mat_->x = *value->vec();
}

Vector3f^ Matrix3f::y::get()
{
    return gcnew Vector3f( new MR::Vector3f( mat_->y ) );
}

void Matrix3f::y::set( Vector3f^ value )
{
    mat_->y = *value->vec();
}

Vector3f^ Matrix3f::z::get()
{
    return gcnew Vector3f( new MR::Vector3f( mat_->z ) );
}

void Matrix3f::z::set( Vector3f^ value )
{
    mat_->z = *value->vec();
}

MR_DOTNET_NAMESPACE_END
