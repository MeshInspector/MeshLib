#include "MRBox3.h"
#include "MRVector3.h"
#pragma managed( push, off )
#include <MRMesh/MRBox.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

Box3f::Box3f()
{
    box_ = new MR::Box3f();
}

Box3f::Box3f( Vector3f^ min, Vector3f^ max )
{
    if ( !min )
        throw gcnew System::ArgumentNullException( "min" );
    if ( !max )
        throw gcnew System::ArgumentNullException( "max" );

    box_ = new MR::Box3f( *min->vec(), *max->vec() );
}

Box3f::Box3f( MR::Box3f* box )
{
    box_ = box;
}

Box3f::~Box3f()
{
    delete box_;
}

Box3f^ Box3f::fromMinAndSize( Vector3f^ min, Vector3f^ size )
{
    if ( !min )
        throw gcnew System::ArgumentNullException( "min" );
    if ( !size )
        throw gcnew System::ArgumentNullException( "size" );

    return gcnew Box3f( new MR::Box3f( std::move( MR::Box3f::fromMinAndSize( *min->vec(), *size->vec() ) ) ) );
}

Vector3f^ Box3f::Min::get()
{
    if ( !min_)
        min_ = gcnew Vector3f( &box_->min );

    return min_;
}

void Box3f::Min::set( Vector3f^ value )
{
    if ( !value )
        throw gcnew System::ArgumentNullException( "value" );

    box_->min = *value->vec();
    delete min_;
}

Vector3f^ Box3f::Max::get()
{
    if ( !max_ )
        max_ = gcnew Vector3f( &box_->max );

    return max_;
}

void Box3f::Max::set( Vector3f^ value )
{
    if ( !value )
        throw gcnew System::ArgumentNullException( "value" );

    box_->max = *value->vec();
    delete max_;
}

Vector3f^ Box3f::Center()
{
    return gcnew Vector3f( new MR::Vector3f( std::move( box_->center() ) ) );
}

Vector3f^ Box3f::Size()
{
    return gcnew Vector3f( new MR::Vector3f( std::move( box_->size() ) ) );
}

float Box3f::Volume()
{
    return box_->volume();
}

float Box3f::Diagonal()
{
    return box_->diagonal();
}

bool Box3f::Valid()
{
    return box_->valid();
}

MR_DOTNET_NAMESPACE_END
