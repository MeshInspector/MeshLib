#pragma once
#include "MRCudaBasic.cuh"
#include "MRCudaMath.cuh"
#include "MRCudaFloat.cuh"

namespace MR
{

namespace Cuda
{
// GPU analog of CPU Dipole struct
struct Dipole
{
    float3 pos;
    float area = 0;
    float3 dirArea;
    float rr = 0; // maximum squared distance from pos to any corner of the bounding box

    /// returns true if this dipole is good approximation for a point \param q;
    /// and adds the contribution of this dipole to the winding number at point \param q to \param addTo
    __device__ bool addIfGoodApprox( const float3& q, float betaSq, float& addTo ) const
    {
        const auto dp = pos - q;
        const auto dd = lengthSq( dp );
        if ( dd <= betaSq * rr )
            return false;
        if ( const auto d = std::sqrt( dd ); d > 0 )
            addTo += dot( dp, dirArea ) / ( d * dd );
        return true;
    }
};

// calls fast winding number for each point in parallel
void fastWindingNumberFromVector( const float3* points, const Dipole* dipoles,
                           const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                           float* resVec, float beta, int skipFace, size_t size );

// calls fast winding number for each triangle center
void fastWindingNumberFromMesh( const Dipole* dipoles,
                                      const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                                      float* resVec, float beta, size_t size );

// calls fast winding number for each point in three-dimensional grid
void fastWindingNumberFromGrid( int3 gridSize, Matrix4 gridToMeshXf,
                                      const Dipole* dipoles, const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                                      float* resVec, float beta );

/// calls fast winding number for each point in three-dimensional grid to get sign
void signedDistance( int3 gridSize, Matrix4 gridToMeshXf,
                                      const Dipole* dipoles, const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                                      float* resVec, float windingNumberThreshold, float beta, float maxDistSq, float minDistSq );


} //namespece Cuda

} //namespace MR
