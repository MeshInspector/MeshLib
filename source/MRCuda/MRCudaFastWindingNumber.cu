#include "MRCudaFastWindingNumber.cuh"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRConstants.h"
#include "device_launch_parameters.h"

namespace MR
{
namespace Cuda
{ 
    constexpr float INV_4PI = 1.0f / ( 4 * PI_F );

    __device__ float Dipole::w( const float3& q ) const
    {
        const auto dp = pos() - q;
        const auto d = length( dp );
        return d > 0 ? INV_4PI * dot( dp, dirArea ) / ( d * d * d ) : 0;
    }

    __device__ float triangleSolidAngle( const float3& p, const float3& tri0, const float3& tri1, const float3& tri2 )
    {
        const auto mx = tri0 - p;
        const auto my = tri1 - p;
        const auto mz = tri2 - p;

        const auto x = length( mx );
        const auto y = length( my );
        const auto z = length( mz );

        auto den = x * y * z + dot( mx, my ) * z + dot( my, mz ) * x + dot( mz, mx ) * y;
        return 2 * std::atan2( mx.x * ( my.y * mz.z - my.z * mz.y ) - mx.y * ( my.x * mz.z - my.z * mz.x ) + mx.z * ( my.x * mz.y - my.y * mz.x ), den );
    }

    __device__ void processPoint( const float3& q, float& res, const Dipole* dipoles,
                                  const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                                  float beta, int skipFace = -1 )
    {
        constexpr int MaxStackSize = 32; // to avoid allocations
        int subtasks[MaxStackSize];
        int stackSize = 0;
        subtasks[stackSize++] = 0;

        while ( stackSize > 0 )
        {
            const auto i = subtasks[--stackSize];
            const auto& node = nodes[i];
            const auto& d = dipoles[i];
            if ( d.goodApprox( q, beta ) )
            {
                res += d.w( q );
                continue;
            }
            if ( !node.leaf() )
            {
                // recurse deeper
                subtasks[stackSize++] = node.r; // to look later
                subtasks[stackSize++] = node.l; // to look first
                continue;
            }
            if ( node.leafId() != skipFace )
            {
                const auto faceVerts = faces[node.leafId()];
                res += INV_4PI * triangleSolidAngle( q, meshPoints[faceVerts.verts[0]], meshPoints[faceVerts.verts[1]], meshPoints[faceVerts.verts[2]] );
            }
        }
    }

    __device__ float calcDistance( const float3& pt, const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces, float maxDistSq, float minDistSq )
    {
        float resSq = maxDistSq;
        struct SubTask
        {
            int n;
            float distSq = 0;
        };

        constexpr int MaxStackSize = 32; // to avoid allocations
        SubTask subtasks[MaxStackSize];
        int stackSize = 0;

        auto addSubTask = [&] ( const SubTask& s )
        {
            if ( s.distSq < resSq )
            {
                assert( stackSize < MaxStackSize );
                subtasks[stackSize++] = s;
            }
        };

        auto getSubTask = [&] ( int n )
        {
            const auto box = nodes[n].box;
            float distSq = lengthSq( box.getBoxClosestPointTo( pt ) - pt );
            return SubTask{ n, distSq };
        };

        addSubTask( getSubTask( 0 ) );

        while ( stackSize > 0 )
        {
            const auto s = subtasks[--stackSize];
            const auto& node = nodes[s.n];
            if ( s.distSq >= resSq )
                continue;

            if ( node.leaf() )
            {
                const auto face = node.leafId();
                const auto& vs = faces[face].verts;
                float3 a = meshPoints[vs[0]];
                float3 b = meshPoints[vs[1]];
                float3 c = meshPoints[vs[2]];

                // compute the closest point in double-precision, because float might be not enough
                const auto closestPointRes = closestPointInTriangle( pt, a, b, c );

                float distSq = lengthSq( closestPointRes.proj - pt );
                if ( distSq < resSq )
                    resSq = distSq;
                if ( distSq <= minDistSq )
                    break;
                continue;
            }

            auto s1 = getSubTask( node.l );
            auto s2 = getSubTask( node.r );
            if ( s1.distSq < s2.distSq )
            {
                const auto temp = s1;
                s1 = s2;
                s2 = temp;
            }
            assert( s1.distSq >= s2.distSq );
            addSubTask( s1 ); // larger distance to look later
            addSubTask( s2 ); // smaller distance to look first
        }
        return sqrt( resSq );
    }

    __global__ void kernel( const float3* points, const Dipole* dipoles,
                            const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                            float* resVec, float beta, int skipFace, size_t size )
    {
        if ( size == 0 )
        {
            assert( false );
            return;
        }

        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if ( index >= size )
            return;

        processPoint( points[index], resVec[index], dipoles, nodes, meshPoints, faces, beta, skipFace );
    }

    __global__ void kernel( const Dipole* dipoles,
                            const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                            float* resVec, float beta, size_t size )
    {
        if ( size == 0 )
        {
            assert( false );
            return;
        }

        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if ( index >= size )
            return;        

        const auto& face = faces[index];
        if ( face.verts[0] < 0 || face.verts[1] < 0 || face.verts[2] < 0 )
            return;

        const auto q = ( meshPoints[face.verts[0]] + meshPoints[face.verts[1]] + meshPoints[face.verts[2]] ) / 3.0f;
        processPoint( q, resVec[index], dipoles, nodes, meshPoints, faces, beta, index );
    }

    __global__ void kernel( int3 dims, float3 minCoord, float3 voxelSize, Matrix4 gridToMeshXf,
                            const Dipole* dipoles, const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                            float* resVec, float beta, size_t size )
    {
        if ( size == 0 )
        {
            assert( false );
            return;
        }

        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if ( index >= size )
            return;

        const int sizeXY = dims.x * dims.y;
        const int sumZ = int( index % sizeXY );
        const int3 voxel{ sumZ % dims.x, sumZ / dims.x, int( index / sizeXY ) };
        const float3 point{ ( minCoord.x + voxel.x ) * voxelSize.x, ( minCoord.y + voxel.y ) * voxelSize.y, ( minCoord.z + voxel.z ) * voxelSize.z };
        const float3 transformedPoint = gridToMeshXf.isIdentity ? point : gridToMeshXf.transform( point );

        processPoint( transformedPoint, resVec[index], dipoles, nodes, meshPoints, faces, beta, index );
    }

    __global__ void kernelWithDistances( int3 dims, float3 minCoord, float3 voxelSize, Matrix4 gridToMeshXf,
                            const Dipole* dipoles, const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                            float* resVec, float beta, float maxDistSq, float minDistSq, size_t size )
    {
        if ( size == 0 )
        {
            assert( false );
            return;
        }

        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if ( index >= size )
            return;

        const int sizeXY = dims.x * dims.y;
        const int sumZ = int( index % sizeXY );
        const int3 voxel{ sumZ % dims.x, sumZ / dims.x, int( index / sizeXY ) };
        const float3 point{ ( minCoord.x + voxel.x ) * voxelSize.x, ( minCoord.y + voxel.y ) * voxelSize.y, ( minCoord.z + voxel.z ) * voxelSize.z };
        const float3 transformedPoint = gridToMeshXf.isIdentity ? point : gridToMeshXf.transform( point );

        float& res = resVec[index];
        res = calcDistance( transformedPoint, nodes, meshPoints, faces, maxDistSq, minDistSq );

        float fwn{ 0 };
        processPoint( transformedPoint, fwn, dipoles, nodes, meshPoints, faces, beta, index );
        if ( fwn > 0.5f )
            res = -res;
    }

    void fastWindingNumberFromVectorKernel( const float3* points, const Dipole* dipoles,
                                  const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                                  float* resVec, float beta, int skipFace, size_t size )
    {
        int maxThreadsPerBlock = 0;
        cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
        int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
        kernel << <numBlocks, maxThreadsPerBlock >> > ( points, dipoles, nodes, meshPoints, faces, resVec, beta, skipFace, size );
    }

    void fastWindingNumberFromMeshKernel( const Dipole* dipoles,
                                          const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                                          float* resVec, float beta, size_t size )
    {
        int maxThreadsPerBlock = 0;
        cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
        int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
        kernel << <numBlocks, maxThreadsPerBlock >> > ( dipoles, nodes, meshPoints, faces, resVec, beta, size );
    }
    void fastWindingNumberFromGridKernel( int3 dims, float3 minCoord, float3 voxelSize, Matrix4 gridToMeshXf,
                                          const Dipole* dipoles, const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                                          float* resVec, float beta )
    {
        const size_t size = size_t( dims.x ) * dims.y * dims.z;
        int maxThreadsPerBlock = 0;
        cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
        int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
        kernel << <numBlocks, maxThreadsPerBlock >> > ( dims, minCoord, voxelSize, gridToMeshXf, dipoles, nodes, meshPoints, faces, resVec, beta, size );       
    }

    void signedDistanceKernel( int3 dims, float3 minCoord, float3 voxelSize, Matrix4 gridToMeshXf,
                                          const Dipole* dipoles, const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                                          float* resVec, float beta, float maxDistSq, float minDistSq )
    {
        const size_t size = size_t( dims.x ) * dims.y * dims.z;
        int maxThreadsPerBlock = 0;
        cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
        int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
        kernelWithDistances << <numBlocks, maxThreadsPerBlock >> > ( dims, minCoord, voxelSize, gridToMeshXf, dipoles, nodes, meshPoints, faces, resVec, beta, maxDistSq, minDistSq, size );
    }
}
}