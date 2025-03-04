#include "MRCudaPointCloud.h"
#include "MRCudaPointCloud.cuh"

#include "MRMesh/MRAABBTreePoints.h"
#include "MRMesh/MRPointCloud.h"

namespace MR::Cuda
{

Expected<std::unique_ptr<PointCloudDataHolder>> copyDataFrom( const PointCloud& pc, bool copyNormals,
    const std::vector<Vector3f>* normals )
{
    const auto& tree = pc.getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& points = tree.orderedPoints();

    auto result = std::make_unique<PointCloudDataHolder>();

    CUDA_LOGE_RETURN_UNEXPECTED( result->nodes.fromVector( nodes.vec_ ) );
    CUDA_LOGE_RETURN_UNEXPECTED( result->points.fromVector( points ) );
    if ( copyNormals )
        CUDA_LOGE_RETURN_UNEXPECTED( result->normals.fromVector( normals ? *normals : pc.normals.vec_ ) );

    return result;
}

size_t pointCloudHeapBytes( const PointCloud& pc, bool copyNormals, const std::vector<Vector3f>* normals )
{
    const auto& tree = pc.getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& points = tree.orderedPoints();

    return
          nodes.size() * sizeof( Node3 )
        + points.size() * sizeof( OrderedPoint )
        + copyNormals ? ( ( normals ? normals->size() : pc.normals.size() ) * sizeof( float3 ) ) : 0
    ;
}

} // namespace MR::Cuda
