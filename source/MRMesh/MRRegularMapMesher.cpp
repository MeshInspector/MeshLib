#include "MRRegularMapMesher.h"
#include "MRRegularGridMesh.h"

namespace MR
{

void RegularMapMesher::setSurfacePC( const std::shared_ptr<PointCloud>& surfacePC )
{
    surfacePC_ = surfacePC;
}

void RegularMapMesher::setDirectionsPC( const std::shared_ptr<PointCloud>& directionsPC )
{
    directionsPC_ = directionsPC;
}

Expected<void> RegularMapMesher::loadDistances( int width, int height, const std::filesystem::path& path )
{
    width_ = width;
    height_ = height;
    std::error_code ec;
    if ( std::filesystem::file_size( path, ec ) != ( height_ * width_ * sizeof( float ) ) )
    {
        distances_.clear();
        return unexpected( "Distances file size is not equal height * width * sizeof(float)" );
    }
    std::ifstream ifs( path, std::ios::binary );
    distances_.resize( width_ );
    ifs.read( (char*) distances_.data(), distances_.size() * sizeof( float ) );
    return {};
}

void RegularMapMesher::setDistances( int width, int height, const std::vector<float>& distances )
{
    width_ = width;
    height_ = height;
    distances_ = distances;
}

Expected<Mesh> RegularMapMesher::createMesh() const
{
    auto refSize = width_ * height_;
    if ( !surfacePC_ )
        return unexpected( "Surface Point Cloud is not loaded" );
    if ( surfacePC_->points.size() != refSize )
        return unexpected( "Surface Point Cloud size is not equal width*height" );

    if ( !directionsPC_ )
        return unexpected( "Directions Point Cloud is not loaded" );
    if ( directionsPC_->points.size() != width_ )
        return unexpected( "Directions Point Cloud size is not equal width" );

    if ( distances_.empty() )
        return unexpected( "Distances file is not loaded" );
    if ( distances_.size() != refSize )
        return unexpected( "Distances size is not equal width*height" );

    auto mesh = makeRegularGridMesh( width_, height_, [&] ( size_t x, size_t y )
    {
        return distances_[x + y * width_] != 0.0;
    },
                                [&] ( size_t x, size_t y )
    {
        VertId idx = VertId( x + y * width_ );
        Vector3f org = surfacePC_->points[idx];
        Vector3f dest = directionsPC_->points[VertId( x )];
        return org + ( dest - org ).normalized() * ( 1.0f / distances_[idx] );
    } );

    if( mesh )
        mesh.value().topology.flipOrientation();

    return mesh;
}

}