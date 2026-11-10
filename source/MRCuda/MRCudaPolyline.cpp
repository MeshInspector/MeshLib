#include "MRCudaPolyline.h"
#include "MRCudaAABBTree.hpp"
#include "MRCudaPolyline.cuh"

#include "MRMesh/MRAABBTreePolyline.h"
#include "MRMesh/MRPolyline.h"

namespace MR::Cuda
{

Expected<Polyline2DataHolder> Polyline2DataHolder::fromLines( const Polyline2& polyline )
{
    const auto& tree = polyline.getAABBTree();
    const auto& points = polyline.points;
    const auto orgs = polyline.topology.getOrgs();

    Polyline2DataHolder result;
    result.root_ = rootNodeId( tree );
    CUDA_LOGE_RETURN_UNEXPECTED( result.nodes_.fromVector( tree.nodes() ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result.leaves_.fromVector( tree.leaves() ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result.points_.fromVector( points ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result.orgs_.fromVector( orgs ) );
    return result;
}

Polyline2Data Polyline2DataHolder::data() const
{
    return {
        .nodes = nodes_.data(),
        .leaves = leaves_.data(),
        .root = root_,
        .points = points_.data(),
        .orgs = orgs_.data(),
    };
}

void Polyline2DataHolder::reset()
{
    nodes_.resize( 0 );
    leaves_.resize( 0 );
    points_.resize( 0 );
    orgs_.resize( 0 );
}

size_t Polyline2DataHolder::heapBytes( const Polyline2& polyline )
{
    return sizeof( Node2 )  * polyline.getAABBTree().nodes().size()
         + sizeof( LineLeaf2 ) * polyline.getAABBTree().numLeaves()
         + sizeof( float2 ) * polyline.points.size()
         + sizeof( int )    * polyline.topology.edgeSize();
}

Expected<Polyline3DataHolder> Polyline3DataHolder::fromLines( const Polyline3& polyline )
{
    const auto& tree = polyline.getAABBTree();
    const auto& points = polyline.points;
    const auto orgs = polyline.topology.getOrgs();

    Polyline3DataHolder result;
    result.root_ = rootNodeId( tree );
    CUDA_LOGE_RETURN_UNEXPECTED( result.nodes_.fromVector( tree.nodes() ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result.leaves_.fromVector( tree.leaves() ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result.points_.fromVector( points ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result.orgs_.fromVector( orgs ) );
    return result;
}

Polyline3Data Polyline3DataHolder::data() const
{
    return {
        .nodes = nodes_.data(),
        .leaves = leaves_.data(),
        .root = root_,
        .points = points_.data(),
        .orgs = orgs_.data(),
    };
}

void Polyline3DataHolder::reset()
{
    nodes_.resize( 0 );
    leaves_.resize( 0 );
    points_.resize( 0 );
    orgs_.resize( 0 );
}

size_t Polyline3DataHolder::heapBytes( const Polyline3& polyline )
{
    return sizeof( TreeNode3 ) * polyline.getAABBTree().nodes().size()
         + sizeof( LineLeaf3 ) * polyline.getAABBTree().numLeaves()
         + sizeof( float3 ) * polyline.points.size()
         + sizeof( int )    * polyline.topology.edgeSize();
}

} // namespace MR::Cuda
