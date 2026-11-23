#include "MRCudaPointCloud.h"
#include "MRCudaPointCloud.cuh"

#include "MRMesh/MRAABBTreePoints.h"
#include "MRMesh/MRPointCloud.h"

#include <cstring>

namespace MR::Cuda
{

static_assert( sizeof( SimdNode3 ) == sizeof( AABBTreePoints::SimdNode ) );
static_assert( sizeof( OrderedPoint ) == sizeof( AABBTreePoints::Point ) );
static_assert( sizeof( Box3 ) == sizeof( Box3f ) );

Expected<std::unique_ptr<PointCloudDataHolder>> copyDataFrom( const PointCloud& pc, bool copyNormals,
    const VertNormals* normals )
{
    const auto& tree = pc.getAABBTree();
    const auto& points = tree.orderedPoints();

    auto result = std::make_unique<PointCloudDataHolder>();

    CUDA_LOGE_RETURN_UNEXPECTED( result->nodes.fromVector( tree.simdNodes() ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result->points.fromVector( points ) );
    if ( copyNormals )
        CUDA_LOGE_RETURN_UNEXPECTED( result->normals.fromVector( normals ? *normals : pc.normals ) );
    const auto box = tree.getBoundingBox();
    std::memcpy( (void*)&result->rootBox, &box, sizeof( Box3 ) );
    result->rootRef = tree.rootRef();
    result->numPoints = int( points.size() );

    return result;
}

size_t pointCloudHeapBytes( const PointCloud& pc, bool copyNormals, const VertNormals* normals )
{
    const auto& tree = pc.getAABBTree();
    const auto& points = tree.orderedPoints();

    return
          tree.simdNodes().size() * sizeof( SimdNode3 )
        + points.size() * sizeof( OrderedPoint )
        + ( copyNormals ? ( normals ? normals->size() : pc.normals.size() ) * sizeof( float3 ) : 0 )
    ;
}

} // namespace MR::Cuda
