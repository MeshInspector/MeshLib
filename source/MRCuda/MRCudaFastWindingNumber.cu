#include "MRCudaFastWindingNumber.cuh"
#include "MRCudaInplaceStack.cuh"

#include "MRMesh/MRConstants.h"
#include "MRMesh/MRDistanceToMeshOptions.h"

#include <limits>

namespace MR
{

namespace Cuda
{

constexpr int maxThreadsPerBlock = 32;

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
    const WideNode3* __restrict__ nodes, const LeafTri3* __restrict__ leafTris, const float3* __restrict__ meshPoints,
    float beta, int skipFace = -1 )
{
    const float betaSq = beta * beta;
    res = 0;

    // every level adds at most 3 nodes, and the tree of 2^31 triangles has 16 levels
    // only the nodes with not good enough approximation by their dipoles get in the stack
    InplaceStack<int, 64> subtasks;
    if ( !dipoles[0].addIfGoodApprox( q, betaSq, res ) )
        subtasks.push( 0 );

    while ( !subtasks.empty() )
    {
        const auto& node = nodes[subtasks.top()];
        subtasks.pop();
        if ( !node.leaf() )
        {
            for ( int c = node.first + node.count - 1; c >= node.first; --c )
                if ( !dipoles[c].addIfGoodApprox( q, betaSq, res ) )
                    subtasks.push( c );
            continue;
        }
        // the triangles of a leaf with not good enough approximation by its dipole are summed exactly
        for ( int t = node.first, e = node.first + node.numLeafTris(); t < e; ++t )
        {
            const auto& lt = leafTris[t];
            if ( lt.face != skipFace )
                res += triangleSolidAngle( q, meshPoints[lt.verts[0]], meshPoints[lt.verts[1]], meshPoints[lt.verts[2]] );
        }
    }
    constexpr float INV_4PI = 1.0f / ( 4 * PI_F );
    res *= INV_4PI;
}

__device__ float calcDistanceSq( const float3& pt,
    const WideNode3* __restrict__ nodes, const LeafTri3* __restrict__ leafTris, const float3* __restrict__ meshPoints,
    float maxDistSq, float minDistSq )
{
    float resSq = maxDistSq;
    struct SubTask
    {
        int n;
        float distSq;
    };

    // every level adds at most 3 tasks, and the tree of 2^31 triangles has 16 levels
    InplaceStack<SubTask, 64> subtasks;

    auto boxDistSq = [&] ( int n )
    {
        return lengthSq( nodes[n].box.getBoxClosestPointTo( pt ) - pt );
    };

    // the node to visit next is not pushed in the stack
    SubTask s{ 0, boxDistSq( 0 ) };
    while ( true )
    {
        if ( s.distSq < resSq )
        {
            const auto& node = nodes[s.n];
            if ( !node.leaf() )
            {
                // the children closer than the found point ordered by distance, the farthest first
                SubTask ch[4];
                int numCh = 0;
                for ( int c = 0; c < node.count; ++c )
                {
                    const int n = node.first + c;
                    const float d = boxDistSq( n );
                    if ( !( d < resSq ) )
                        continue;
                    int k = numCh++;
                    for ( ; k > 0 && ch[k - 1].distSq < d; --k )
                        ch[k] = ch[k - 1];
                    ch[k] = { n, d };
                }
                if ( numCh > 0 )
                {
                    // descend in the nearest child, the others wait in the stack
                    for ( int i = 0; i + 1 < numCh; ++i )
                        subtasks.push( ch[i] );
                    s = ch[numCh - 1];
                    continue;
                }
            }
            else
            {
                for ( int t = node.first, e = node.first + node.numLeafTris(); t < e; ++t )
                {
                    const auto& lt = leafTris[t];
                    const auto closestPointRes = closestPointInTriangle( pt, meshPoints[lt.verts[0]], meshPoints[lt.verts[1]], meshPoints[lt.verts[2]] );
                    const float distSq = lengthSq( closestPointRes.proj - pt );
                    if ( distSq < resSq )
                        resSq = distSq;
                }
                if ( resSq <= minDistSq )
                    break;
            }
        }
        if ( subtasks.empty() )
            break;
        s = subtasks.top();
        subtasks.pop();
    }
    return resSq;
}

__global__ void fastWindingNumberFromVectorKernel( const float3* points,
    const Dipole* __restrict__ dipoles, const WideNode3* __restrict__ nodes, const LeafTri3* __restrict__ leafTris, const float3* __restrict__ meshPoints, const FaceToThreeVerts* __restrict__ faces,
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

    processPoint( points[index], resVec[index], dipoles, nodes, leafTris, meshPoints, beta, skipFace );
}

__global__ void fastWindingNumberFromMeshKernel( const Dipole* __restrict__ dipoles,
    const WideNode3* __restrict__ nodes, const LeafTri3* __restrict__ leafTris, const float3* __restrict__ meshPoints, const FaceToThreeVerts* __restrict__ faces,
    float* resVec, float beta, size_t chunkSize, size_t chunkOffset )
{
    if ( chunkSize == 0 )
    {
        assert( false );
        return;
    }

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= chunkSize )
        return;

    const size_t faceIndex = index + chunkOffset;
    const auto& face = faces[faceIndex];
    if ( face.verts[0] < 0 || face.verts[1] < 0 || face.verts[2] < 0 )
        return;

    const auto q = ( meshPoints[face.verts[0]] + meshPoints[face.verts[1]] + meshPoints[face.verts[2]] ) / 3.0f;
    processPoint( q, resVec[index], dipoles, nodes, leafTris, meshPoints, beta, faceIndex );
}

__global__ void fastWindingNumberFromGridKernel( int3 dims, Matrix4 gridToMeshXf,
    const Dipole* __restrict__ dipoles, const WideNode3* __restrict__ nodes, const LeafTri3* __restrict__ leafTris, const float3* __restrict__ meshPoints, const FaceToThreeVerts* __restrict__ faces,
    float* resVec, float beta, size_t chunkSize, size_t chunkOffset )
{
    if ( chunkSize == 0 )
    {
        assert( false );
        return;
    }

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= chunkSize )
        return;

    const size_t gridIndex = index + chunkOffset;
    const size_t gridSize = size_t( dims.x ) * dims.y * dims.z;
    if ( gridIndex >= gridSize )
        return;

    const int sizeXY = dims.x * dims.y;
    const int sumZ = int( gridIndex % sizeXY );
    const int3 voxel{ sumZ % dims.x, sumZ / dims.x, int( gridIndex / sizeXY ) };
    const float3 point{ float( voxel.x ), float( voxel.y ), float( voxel.z ) };
    const float3 transformedPoint = gridToMeshXf.isIdentity ? point : gridToMeshXf.transform( point );

    processPoint( transformedPoint, resVec[index], dipoles, nodes, leafTris, meshPoints, beta );
}

static constexpr float cQuietNan = std::numeric_limits<float>::quiet_NaN();

__global__ void signedDistanceKernel( int3 dims, Matrix4 gridToMeshXf,
    const Dipole* __restrict__ dipoles, const WideNode3* __restrict__ nodes, const LeafTri3* __restrict__ leafTris, const float3* __restrict__ meshPoints, const FaceToThreeVerts* __restrict__ faces,
    float* resVec, DistanceToMeshOptions options, size_t chunkSize, size_t chunkOffset ) // pass options by value to avoid reference on CPU memory
{
    if ( chunkSize == 0 )
    {
        assert( false );
        return;
    }

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= chunkSize )
        return;

    const size_t gridIndex = index + chunkOffset;
    const size_t gridSize = size_t( dims.x ) * dims.y * dims.z;
    if ( gridIndex >= gridSize )
        return;

    const int sizeXY = dims.x * dims.y;
    const int sumZ = int( gridIndex % sizeXY );
    const int3 voxel{ sumZ % dims.x, sumZ / dims.x, int( gridIndex / sizeXY ) };
    const float3 point{ float( voxel.x ), float( voxel.y ), float( voxel.z ) };
    const float3 transformedPoint = gridToMeshXf.isIdentity ? point : gridToMeshXf.transform( point );

    float resSq = calcDistanceSq( transformedPoint, nodes, leafTris, meshPoints, options.maxDistSq, options.minDistSq );
    if ( options.nullOutsideMinMax && ( resSq < options.minDistSq || resSq >= options.maxDistSq ) ) // note that resSq == minDistSq (e.g. == 0) is a valid situation
    {
        resVec[index] = cQuietNan;
        return;
    }

    float fwn{ 0 };
    processPoint( transformedPoint, fwn, dipoles, nodes, leafTris, meshPoints, options.windingNumberBeta );
    float res = sqrt( resSq );
    if ( fwn > options.windingNumberThreshold )
        res = -res;
    resVec[index] = res;
}

void fastWindingNumberFromVector( const float3* points,
                                FastWindingNumberData data,
                                float* resVec, float beta, int skipFace, size_t size )
{
    int numBlocks = int( ( size + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    fastWindingNumberFromVectorKernel<<< numBlocks, maxThreadsPerBlock >>>( points, data.dipoles, data.nodes, data.leafTris, data.meshPoints, data.faces, resVec, beta, skipFace, size );
}

void fastWindingNumberFromMesh( FastWindingNumberData data,
                                        float* resVec, float beta, size_t chunkSize, size_t chunkOffset )
{
    int numBlocks = int( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    fastWindingNumberFromMeshKernel<<< numBlocks, maxThreadsPerBlock >>>( data.dipoles, data.nodes, data.leafTris, data.meshPoints, data.faces, resVec, beta, chunkSize, chunkOffset );
}

void fastWindingNumberFromGrid( int3 dims, Matrix4 gridToMeshXf,
                                        FastWindingNumberData data,
                                        float* resVec, float beta, size_t chunkSize, size_t chunkOffset )
{
    int numBlocks = int( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    fastWindingNumberFromGridKernel<<< numBlocks, maxThreadsPerBlock >>>( dims, gridToMeshXf, data.dipoles, data.nodes, data.leafTris, data.meshPoints, data.faces, resVec, beta, chunkSize, chunkOffset );
}

void signedDistance( int3 dims, Matrix4 gridToMeshXf,
                     FastWindingNumberData data,
                     float* resVec, size_t chunkSize, size_t chunkOffset, const DistanceToMeshOptions& options )
{
    int numBlocks = int( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    signedDistanceKernel<<< numBlocks, maxThreadsPerBlock >>>( dims, gridToMeshXf, data.dipoles, data.nodes, data.leafTris, data.meshPoints, data.faces, resVec, options, chunkSize, chunkOffset );
}

} //namespace Cuda

} //namespace MR
