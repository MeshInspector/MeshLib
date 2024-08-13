#include "MRPointCloud.h"
#include "MRVector3.h"
#include "MRBox3.h"
#include "MRBitSet.h"
#include "MRAffineXf.h"

#pragma managed( push, off )
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRPointsLoad.h>
#include <MRMesh/MRPointsSave.h>
#include <MRMesh/MRBitSet.h>
#pragma managed( pop )

#include <msclr/marshal_cppstd.h>

MR_DOTNET_NAMESPACE_BEGIN

PointCloud::PointCloud()
{
    pc_ = new MR::PointCloud();
}

PointCloud::PointCloud( MR::PointCloud* pc )
{
    pc_ = pc;
}

PointCloud::~PointCloud()
{
    delete pc_;
}

VertCoordsReadOnly^ PointCloud::Points::get()
{
    if ( !points_ )
    {
        const auto& points = pc_->points;
        points_ = gcnew VertCoords( int( points.size() ) );
        for ( size_t i = 0; i < points.size(); i++ )
            points_->Add( gcnew Vector3f( new MR::Vector3f( points.vec_[i] ) ) );
    }
    return points_->AsReadOnly();
}

VertCoordsReadOnly^ PointCloud::Normals::get()
{
    if ( !normals_ )
    {
        const auto& normals = pc_->normals;
        normals_ = gcnew VertCoords( int( normals.size() ) );
        for ( size_t i = 0; i < normals.size(); i++ )
            normals_->Add( gcnew Vector3f( new MR::Vector3f( normals.vec_[i] ) ) );
    }
    return normals_->AsReadOnly();
}

VertBitSetReadOnly^ PointCloud::ValidPoints::get()
{
    if ( !validPoints_ )
        validPoints_ = gcnew VertBitSet( new MR::VertBitSet( pc_->validPoints ) );

    return validPoints_;
}

Box3f^ PointCloud::BoundingBox::get()
{
    if ( !boundingBox_ )
        boundingBox_ = gcnew Box3f( new MR::Box3f( std::move( pc_->computeBoundingBox() ) ) );

    return boundingBox_;
}

PointCloud^ PointCloud::FromAnySupportedFormat( System::String^ path )
{
    if ( !path )
        throw gcnew System::ArgumentNullException();

    std::filesystem::path nativePath( msclr::interop::marshal_as<std::string>( path ) );
    auto pointsOrErr = MR::PointsLoad::fromAnySupportedFormat( nativePath );

    if ( !pointsOrErr )
        throw gcnew System::SystemException( gcnew System::String( pointsOrErr.error().c_str() ) );

    return gcnew PointCloud( new MR::PointCloud( std::move( *pointsOrErr ) ) );
}

void PointCloud::ToAnySupportedFormat( PointCloud^ pc, System::String^ path )
{
    if ( !pc )
        throw gcnew System::ArgumentNullException( "mesh" );

    if ( !path )
        throw gcnew System::ArgumentNullException( "path" );

    std::filesystem::path nativePath( msclr::interop::marshal_as<std::string>( path ) );
    auto err = MR::PointsSave::toAnySupportedFormat( *pc->pc_, nativePath );

    if ( !err )
    {
        std::string error = err.error();
        throw gcnew System::SystemException( gcnew System::String( error.c_str() ) );
    }
}

void PointCloud::clearManagedResources()
{
    points_ = nullptr;
    normals_ = nullptr;
    validPoints_ = nullptr;
}

void PointCloud::AddPoint( Vector3f^ p )
{
    if ( !p )
        throw gcnew System::ArgumentNullException( "p" );

    if ( !pc_->normals.empty() )
        throw gcnew System::InvalidOperationException( "Normals must be empty" );

    pc_->addPoint( *p->vec() );
    clearManagedResources();
}

void PointCloud::AddPoint( Vector3f^ p, Vector3f^ n )
{
    if ( !p )
        throw gcnew System::ArgumentNullException( "p" );
    if ( !n )
        throw gcnew System::ArgumentNullException( "n" );

    if ( pc_->normals.size() != pc_->points.size() )
        throw gcnew System::InvalidOperationException( "Points and normals must have the same size" );

    pc_->addPoint( *p->vec(), *n->vec() );
    clearManagedResources();
}

MR_DOTNET_NAMESPACE_END
