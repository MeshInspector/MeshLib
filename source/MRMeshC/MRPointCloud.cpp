#include "MRPointCloud.h"

#include "MRMesh/MRPointCloud.h"

#include <span>

using namespace MR;

MRPointCloud* mrPointCloudNew( const MRVector3f* points_, size_t pointsNum )
{
    std::span points { reinterpret_cast<const Vector3f*>( points_ ), pointsNum };

    PointCloud res;
    res.points = { points.begin(), points.end() };
    return reinterpret_cast<MRPointCloud*>( new PointCloud( std::move( res ) ) );
}

const MRVector3f* mrPointCloudPoints( const MRPointCloud* pc_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    return reinterpret_cast<const MRVector3f*>( pc.points.data() );
}

size_t mrPointCloudPointsNum( const MRPointCloud* pc_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    return pc.points.size();
}

const MRVector3f* mrPointCloudNormals( const MRPointCloud* pc_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    return reinterpret_cast<const MRVector3f*>( pc.normals.data() );
}

size_t mrPointCloudNormalsNum( const MRPointCloud* pc_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    return pc.normals.size();
}

const MRVertBitSet* mrPointCloudValidPoints( const MRPointCloud* pc_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    return reinterpret_cast<const MRVertBitSet*>( &pc.validPoints );
}

