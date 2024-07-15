#include "MRPointCloud.h"

#include "MRMesh/MRPointCloud.h"

using namespace MR;

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

