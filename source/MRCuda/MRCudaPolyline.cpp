#include "MRCudaPolyline.h"
#include "MRCudaPolyline.cuh"

#include "MRMesh/MRAABBTreePolyline.h"
#include "MRMesh/MRPolyline.h"

namespace MR::Cuda
{

Expected<Polyline2DataHolder> Polyline2DataHolder::fromLines( const Polyline2& polyline )
{
    const auto& nodes = polyline.getAABBTree().nodes();
    const auto& points = polyline.points;
    const auto orgs = polyline.topology.getOrgs();

    Polyline2DataHolder result;
    CUDA_LOGE_RETURN_UNEXPECTED( result.nodes_.fromVector( nodes ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result.points_.fromVector( points ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result.orgs_.fromVector( orgs ) );
    return result;
}

Polyline2Data Polyline2DataHolder::data() const
{
    return {
        .nodes = nodes_.data(),
        .points = points_.data(),
        .orgs = orgs_.data(),
    };
}

void Polyline2DataHolder::reset()
{
    nodes_.resize( 0 );
    points_.resize( 0 );
    orgs_.resize( 0 );
}

size_t Polyline2DataHolder::heapBytes( const Polyline2& polyline )
{
    return sizeof( Node2 )  * polyline.getAABBTree().nodes().size()
         + sizeof( float2 ) * polyline.points.size()
         + sizeof( int )    * polyline.topology.edgeSize();
}

Expected<Polyline3DataHolder> Polyline3DataHolder::fromLines( const Polyline3& polyline )
{
    const auto& nodes = polyline.getAABBTree().nodes();
    const auto& points = polyline.points;
    const auto orgs = polyline.topology.getOrgs();

    Polyline3DataHolder result;
    CUDA_LOGE_RETURN_UNEXPECTED( result.nodes_.fromVector( nodes ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result.points_.fromVector( points ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result.orgs_.fromVector( orgs ) );
    return result;
}

Polyline3Data Polyline3DataHolder::data() const
{
    return {
        .nodes = nodes_.data(),
        .points = points_.data(),
        .orgs = orgs_.data(),
    };
}

void Polyline3DataHolder::reset()
{
    nodes_.resize( 0 );
    points_.resize( 0 );
    orgs_.resize( 0 );
}

size_t Polyline3DataHolder::heapBytes( const Polyline3& polyline )
{
    return sizeof( Node3 )  * polyline.getAABBTree().nodes().size()
         + sizeof( float3 ) * polyline.points.size()
         + sizeof( int )    * polyline.topology.edgeSize();
}

} // namespace MR::Cuda
