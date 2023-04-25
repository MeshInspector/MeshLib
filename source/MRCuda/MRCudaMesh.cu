#include "MRCudaMesh.cuh"
#include "MRCudaFloat.cuh"

namespace MR
{
namespace Cuda
{

__device__ bool Node3::leaf() const
{
    return r < 0;
}

__device__ int Node3::leafId() const
{
    return l;
}

__device__ float3 Box3::getBoxClosestPointTo( const float3& pt ) const
{
    return { clamp( pt.x, min.x, max.x ), clamp( pt.y, min.y, max.y ), clamp( pt.z, min.z, max.z ) };
}

__device__ void Box3::include( const float3& pt )
{
    if ( pt.x < min.x ) min.x = pt.x;
    if ( pt.x > max.x ) max.x = pt.x;
    if ( pt.y < min.y ) min.y = pt.y;
    if ( pt.y > max.y ) max.y = pt.y;
    if ( pt.z < min.z ) min.z = pt.z;
    if ( pt.z > max.z ) max.z = pt.z;
}
}
}