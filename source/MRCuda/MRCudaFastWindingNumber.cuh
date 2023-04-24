#pragma once
#include "MRCudaBasic.cuh"
#include "MRCudaMesh.cuh"
#include "MRCudaFloat.cuh"

namespace MR
{

namespace Cuda
{

struct Dipole
{
    float3 areaPos;
    float area = 0;
    float3 dirArea;
    float rr = 0; // maximum squared distance from pos to any corner of the bounding box
    __host__ __device__ float3 pos() const
    {
        return area > 0 ? areaPos / area : areaPos;
    }
    /// returns true if this dipole is good approximation for a point \param q
    __host__ __device__ bool goodApprox( const float3& q, float beta ) const
    {
        return lengthSq( q - pos() ) > beta * beta * rr;
    }
    /// contribution of this dipole to the winding number at point \param q
    __host__ __device__ float w( const float3& q ) const;
};

void fastWindingNumberFromVectorKernel( const float3* points, const Dipole* dipoles,
                           const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                           float* resVec, float beta, int skipFace, size_t size );

void fastWindingNumberFromMeshKernel( const Dipole* dipoles,
                                      const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                                      float* resVec, float beta, size_t size );

void fastWindingNumberFromGridKernel( int3 gridSize, float3 minCoord, float3 voxelSize, Matrix4 gridToMeshXf,
                                      const Dipole* dipoles, const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                                      float* resVec, float beta );

}
}
