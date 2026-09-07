#pragma once

#include "MRCuda/MRCudaMath.cuh"
#include "MRCuda/MRCudaPolyline.cuh"

namespace MR::Cuda
{

struct VolumeIndexer
{
    int3 dims;
    float voxelSize;
    float3 origin;

    VolumeIndexer( int3 dims, float voxelSize, float3 origin ) : dims( dims ), voxelSize( voxelSize ), origin( origin )
    {
        sizeXY_ = std::uint64_t( dims.x ) * dims.y;
    }

    __host__ __device__ inline std::uint64_t size() const
    {
        return sizeXY_ * dims.z;
    }

    __host__ __device__ inline int3 toCoord( int vox ) const
    {
        int3 result;
        result.z = int( vox / sizeXY_ );
        const auto xy = vox % sizeXY_;
        result.y = int( xy / dims.x );
        result.x = int( xy % dims.x );
        return result;
    }

    __host__ __device__ inline float3 toPoint( int3 coord ) const
    {
        return float3 {
            ( (float)coord.x + 0.5f ) * voxelSize + origin.x,
            ( (float)coord.y + 0.5f ) * voxelSize + origin.y,
            ( (float)coord.z + 0.5f ) * voxelSize + origin.z
        };
    }

private:
    std::uint64_t sizeXY_;
};

struct FlatEndMillTool
{
    float length;
    float radius;
};

struct BallEndMillTool
{
    float length;
    float radius;
};

struct BullNoseEndMillTool
{
    float length;
    float radius;
    float cornerRadius;
};

struct ChamferEndMillTool
{
    float length;
    float radius;
    float endRadius;
    float cutterHeight;
};

void computeToolDistanceKernel(
    float* __restrict__ output, size_t size,
    VolumeIndexer indexer,
    Polyline3Data toolpath,
    Polyline2Data toolPolyline,
    float toolRadius, float toolMinHeight, float toolMaxHeight,
    float padding
);

void computeToolDistanceKernel(
    float* __restrict__ output, size_t size,
    VolumeIndexer indexer,
    Polyline3Data toolpath,
    FlatEndMillTool tool,
    float padding
);

void computeToolDistanceKernel(
    float* __restrict__ output, size_t size,
    VolumeIndexer indexer,
    Polyline3Data toolpath,
    BallEndMillTool tool,
    float padding
);

void computeToolDistanceKernel(
    float* __restrict__ output, size_t size,
    VolumeIndexer indexer,
    Polyline3Data toolpath,
    BullNoseEndMillTool tool,
    float padding
);

void computeToolDistanceKernel(
    float* __restrict__ output, size_t size,
    VolumeIndexer indexer,
    Polyline3Data toolpath,
    ChamferEndMillTool tool,
    float padding
);

} // namespace MR::Cuda
