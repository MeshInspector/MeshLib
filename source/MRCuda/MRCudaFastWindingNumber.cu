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
    const MeshTree& tree, const float3* __restrict__ meshPoints,
    float beta, int skipFace = -1 )
{
    const float betaSq = beta * beta;

    InplaceStack<int, 32> subtasks;
    subtasks.push( tree.root );

    while ( !subtasks.empty() )
    {
        const auto i = subtasks.top();
        subtasks.pop();
        if ( MeshTree::isLeaf( i ) )
        {
            const auto& leaf = tree.leaf( i );
            if ( leaf.face != skipFace )
                res += triangleSolidAngle( q, meshPoints[leaf.verts[0]], meshPoints[leaf.verts[1]], meshPoints[leaf.verts[2]] );
            continue;
        }
        if ( dipoles[i].addIfGoodApprox( q, betaSq, res ) )
            continue;
        // recurse deeper
        const auto& node = tree.nodes[i];
        subtasks.push( node.r ); // to look later
        subtasks.push( node.l ); // to look first
    }
    constexpr float INV_4PI = 1.0f / ( 4 * PI_F );
    res *= INV_4PI;
}

__device__ float calcDistanceSq( const float3& pt,
    const MeshTree& tree, const float3* __restrict__ meshPoints,
    float maxDistSq, float minDistSq )
{
    float resSq = maxDistSq;
    struct SubTask
    {
        int n;
        float distSq;
    };

    InplaceStack<SubTask, 32> subtasks;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < resSq )
            subtasks.push( s );
    };

    auto getSubTask = [&] ( int n )
    {
        const auto box = tree.box( n );
        float distSq = lengthSq( box.getBoxClosestPointTo( pt ) - pt );
        return SubTask{ n, distSq };
    };

    addSubTask( getSubTask( tree.root ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( s.distSq >= resSq )
            continue;

        if ( MeshTree::isLeaf( s.n ) )
        {
            const auto& vs = tree.leaf( s.n ).verts;
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

        const auto& node = tree.nodes[s.n];
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
    return resSq;
}

__global__ void fastWindingNumberFromVectorKernel( const float3* points,
    const Dipole* __restrict__ dipoles, const MeshTree tree, const float3* __restrict__ meshPoints,
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

    processPoint( points[index], resVec[index], dipoles, tree, meshPoints, beta, skipFace );
}

__global__ void fastWindingNumberFromMeshKernel( const Dipole* __restrict__ dipoles,
    const MeshTree tree, const float3* __restrict__ meshPoints, const FaceToThreeVerts* __restrict__ faces,
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
    processPoint( q, resVec[index], dipoles, tree, meshPoints, beta, faceIndex );
}

__global__ void fastWindingNumberFromGridKernel( int3 dims, Matrix4 gridToMeshXf,
    const Dipole* __restrict__ dipoles, const MeshTree tree, const float3* __restrict__ meshPoints,
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

    processPoint( transformedPoint, resVec[index], dipoles, tree, meshPoints, beta );
}

static constexpr float cQuietNan = std::numeric_limits<float>::quiet_NaN();

__global__ void signedDistanceKernel( int3 dims, Matrix4 gridToMeshXf,
    const Dipole* __restrict__ dipoles, const MeshTree tree, const float3* __restrict__ meshPoints,
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

    float resSq = calcDistanceSq( transformedPoint, tree, meshPoints, options.maxDistSq, options.minDistSq );
    if ( options.nullOutsideMinMax && ( resSq < options.minDistSq || resSq >= options.maxDistSq ) ) // note that resSq == minDistSq (e.g. == 0) is a valid situation
    {
        resVec[index] = cQuietNan;
        return;
    }

    float fwn{ 0 };
    processPoint( transformedPoint, fwn, dipoles, tree, meshPoints, options.windingNumberBeta );
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
    fastWindingNumberFromVectorKernel<<< numBlocks, maxThreadsPerBlock >>>( points, data.dipoles, data.tree, data.meshPoints, resVec, beta, skipFace, size );
}

void fastWindingNumberFromMesh( FastWindingNumberData data,
                                        float* resVec, float beta, size_t chunkSize, size_t chunkOffset )
{
    int numBlocks = int( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    fastWindingNumberFromMeshKernel<<< numBlocks, maxThreadsPerBlock >>>( data.dipoles, data.tree, data.meshPoints, data.faces, resVec, beta, chunkSize, chunkOffset );
}

void fastWindingNumberFromGrid( int3 dims, Matrix4 gridToMeshXf,
                                        FastWindingNumberData data,
                                        float* resVec, float beta, size_t chunkSize, size_t chunkOffset )
{
    int numBlocks = int( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    fastWindingNumberFromGridKernel<<< numBlocks, maxThreadsPerBlock >>>( dims, gridToMeshXf, data.dipoles, data.tree, data.meshPoints, resVec, beta, chunkSize, chunkOffset );
}

void signedDistance( int3 dims, Matrix4 gridToMeshXf,
                     FastWindingNumberData data,
                     float* resVec, size_t chunkSize, size_t chunkOffset, const DistanceToMeshOptions& options )
{
    int numBlocks = int( ( chunkSize + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    signedDistanceKernel<<< numBlocks, maxThreadsPerBlock >>>( dims, gridToMeshXf, data.dipoles, data.tree, data.meshPoints, resVec, options, chunkSize, chunkOffset );
}

} //namespace Cuda

} //namespace MR
