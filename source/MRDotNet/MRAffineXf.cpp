#include "MRAffineXf.h"
#include "MRMatrix3.h"
#include "MRVector3.h"

#pragma managed( push, off )
#include <MRMesh/MRAffineXf.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRVector3.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

AffineXf3f::AffineXf3f()
    : AffineXf3f( gcnew Matrix3f(), gcnew Vector3f() )
{
}

AffineXf3f::AffineXf3f( Matrix3f^ A )
    : AffineXf3f( A, gcnew Vector3f() )
{
}

AffineXf3f::AffineXf3f( Vector3f^ b )
    : AffineXf3f( gcnew Matrix3f(), b )
{
}

AffineXf3f::AffineXf3f( Matrix3f^ A, Vector3f^ b )
{
    if ( !A || !b )
        throw gcnew System::ArgumentNullException();

    xf_ = new MR::AffineXf3f( *A->mat(), *b->vec() );
    this->A = A;
    this->b = b;
}

AffineXf3f::AffineXf3f( MR::AffineXf3f* xf )
{
    xf_ = xf;
    this->A = gcnew Matrix3f( new MR::Matrix3f( xf_->A ) );
    this->b = gcnew Vector3f( new MR::Vector3f( xf_->b ) );
}

AffineXf3f::~AffineXf3f()
{
    delete xf_;
}

AffineXf3f^ AffineXf3f::operator*( AffineXf3f^ a, AffineXf3f^ b )
{
    if ( !a || !b )
        throw gcnew System::ArgumentNullException();

    return gcnew AffineXf3f( new MR::AffineXf3f( *a->xf_ * *b->xf_ ) );
}

Matrix3f^ AffineXf3f::A::get()
{
    return A_;
}

void AffineXf3f::A::set( Matrix3f^ value )
{
    if ( !value )
        throw gcnew System::ArgumentNullException();

    xf_->A = *value->mat();
    A_ = value;
}

Vector3f^ AffineXf3f::b::get()
{
    return b_;
}

void AffineXf3f::b::set( Vector3f^ value )
{
    if ( !value )
        throw gcnew System::ArgumentNullException();

    xf_->b = *value->vec();
    b_ = value;
}

MR_DOTNET_NAMESPACE_END
