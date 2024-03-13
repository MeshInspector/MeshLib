#include "MRCudaPointsToMeshFusion.h"
#include "MRCudaPointsToMeshFusion.cuh"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRAABBTreePoints.h"

namespace MR
{
namespace Cuda
{
    Expected<MR::SimpleVolume> pointsToDistanceVolume( const PointCloud& cloud, const MR::PointsToDistanceVolumeParams& params )
    {
        const auto& tree = cloud.getAABBTree();
        const auto& nodes = tree.nodes();

        DynamicArray<Node3> cudaNodes;
        cudaNodes.fromVector( nodes.vec_ );
        
        DynamicArray<OrderedPoint> cudaPoints;
        cudaPoints.fromVector( tree.orderedPoints() );

        DynamicArray<float3> cudaNormals;
        cudaNormals.fromVector( cloud.normals.vec_ );

        PointsToDistanceVolumeParams cudaParams
        {            
            .sigma = params.sigma,
            .minWeight = params.minWeight
        };

        cudaParams.origin.x = params.origin.x;
        cudaParams.origin.y = params.origin.y;
        cudaParams.origin.z = params.origin.z;

        cudaParams.voxelSize.x = params.voxelSize.x;
        cudaParams.voxelSize.y = params.voxelSize.y;
        cudaParams.voxelSize.z = params.voxelSize.z;

        cudaParams.dimensions.x = params.dimensions.x;
        cudaParams.dimensions.y = params.dimensions.y;
        cudaParams.dimensions.z = params.dimensions.z;


        auto box = cloud.getBoundingBox();
        auto expansion = 2.0f * params.voxelSize;
       
        DynamicArray<float> cudaVolume;
        cudaVolume.resize( params.dimensions.x * params.dimensions.y * params.dimensions.z );

        /*std::vector<Node3> cudaNodesVec;
        cudaNodes.toVector( cudaNodesVec );

        std::vector<OrderedPoint> cudaPointsVec;
        cudaPoints.toVector( cudaPointsVec );

        std::vector<float3> cudaNormalsVec;
        cudaNormals.toVector( cudaNormalsVec );

        std::vector<float> cudaVolumeVec;
        cudaVolume.toVector( cudaVolumeVec );

        for ( int id = 0; id < cudaVolume.size(); ++id )
        {
            unsigned char quietNan[4] = { 0x7f , 0xc0,  0x00, 0x00 };
            cudaVolumeVec[id] = *( float* )quietNan;

            const int sizeXY = params.dimensions.x * params.dimensions.y;
            float3 coord;
            coord.z = int( id / sizeXY ) + 0.5f;
            int sumZ = int( id % sizeXY );
            coord.y = sumZ / params.dimensions.x + 0.5f;
            coord.x = sumZ % params.dimensions.x + 0.5f;

            float3 voxelCenter = cudaParams.origin;
            voxelCenter.x += cudaParams.voxelSize.x * coord.x;
            voxelCenter.y += cudaParams.voxelSize.y * coord.y;
            voxelCenter.z += cudaParams.voxelSize.z * coord.z;

            float sumDist = 0;
            float sumWeight = 0;

            const float radius = 3 * cudaParams.sigma;
            const float radiusSq = radius * radius;

            constexpr int MaxStackSize = 32; // to avoid allocations
            int subtasks[MaxStackSize];
            int stackSize = 0;
            subtasks[stackSize++] = 0;

            auto addSubTask = [&] ( int n )
            {
                float distSq = lengthSq( cudaNodesVec[n].box.getBoxClosestPointTo( voxelCenter ) - voxelCenter );
                if ( distSq <= radiusSq )
                    subtasks[stackSize++] = n;
            };

            addSubTask( 0 );
            const auto inv2SgSq = -0.5f / ( params.sigma * params.sigma );
            while ( stackSize > 0 )
            {
                const auto n = subtasks[--stackSize];
                const auto& node = cudaNodesVec[n];

                if ( node.leaf() )
                {
                    auto [first, last] = node.getLeafPointRange();
                    for ( int i = first; i < last; ++i )
                    {
                        //auto coord = cudaPointsVec[i].coord;
                        if ( lengthSq( cudaPointsVec[i].coord - voxelCenter ) <= radiusSq )
                        {
                            const auto distSq = lengthSq( voxelCenter - cudaPointsVec[i].coord );
                            const auto w = exp( distSq * inv2SgSq );
                            sumWeight += w;
                            sumDist += dot( cudaNormalsVec[cudaPointsVec[i].id], voxelCenter - cudaPointsVec[i].coord ) * w;
                        }
                        //foundCallback( orderedPoints[i].id, coord );
                    }
                    continue;
                }

                addSubTask( node.r ); // look at right node later
                addSubTask( node.l ); // look at left node first
            }

            if ( sumWeight >= params.minWeight )
                cudaVolumeVec[id] = sumDist / sumWeight;
        }
        return {};*/
        pointsToDistanceVolumeKernel(cudaNodes.data(), cudaPoints.data(), cudaNormals.data(), cudaVolume.data(), cudaParams);

        MR::SimpleVolume res;
        res.dims = params.dimensions;
        res.voxelSize = params.voxelSize;
        res.max = params.sigma * std::exp( -0.5f );
        res.min = -res.max;
        cudaVolume.toVector( res.data );
        return res;
    }
}
}