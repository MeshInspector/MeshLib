#include "MRCudaPointsToMeshProjector.cuh"
#include "MRCudaInplaceStack.cuh"

namespace MR { namespace Cuda {

__global__ void kernel( const float3* points,
    const WideNode3* __restrict__ nodes, const float3* __restrict__ meshPoints, const LeafTri3* __restrict__ leafTris,
    MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size )
{
    if ( size == 0 )
    {
        assert( false );
        return;
    }

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= size )
        return;

    const auto pt = xf.isIdentity ? points[index] : xf.transform( points[index] );
    MeshProjectionResult res;
    res.distSq = upDistLimitSq;
    res.proj.faceId = -1;
    struct SubTask
    {
        int n;
        float distSq;
    };

    // every level adds at most 3 tasks, and the tree of 2^31 triangles has 16 levels
    InplaceStack<SubTask, 64> subtasks;

    auto boxDistSq = [&] ( int n )
    {
        const auto box = refXf.isIdentity ? nodes[n].box : refXf.transform( nodes[n].box );
        return lengthSq( box.getBoxClosestPointTo( pt ) - pt );
    };

    subtasks.push( { 0, boxDistSq( 0 ) } );

    bool stop = false;
    while ( !stop && !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( !( s.distSq < res.distSq ) )
            continue;

        const auto& node = nodes[s.n];
        if ( node.leaf() )
        {
            for ( int i = node.first, e = node.first + node.numLeafTris(); i < e; ++i )
            {
                const auto& lt = leafTris[i];
                float3 a = meshPoints[lt.verts[0]];
                float3 b = meshPoints[lt.verts[1]];
                float3 c = meshPoints[lt.verts[2]];

                if ( !refXf.isIdentity )
                {
                    a = refXf.transform( a );
                    b = refXf.transform( b );
                    c = refXf.transform( c );
                }

                const auto closestPointRes = closestPointInTriangle( pt, a, b, c );

                float distSq = lengthSq( closestPointRes.proj - pt );
                if ( distSq < res.distSq )
                {
                    res.distSq = distSq;
                    res.proj.point = closestPointRes.proj;
                    res.proj.faceId = lt.face;
                    res.tp = MeshTriPoint{ -1, closestPointRes.bary.x, closestPointRes.bary.y };
                    if ( distSq <= loDistLimitSq )
                    {
                        stop = true;
                        break;
                    }
                }
            }
            continue;
        }

        // push the children closer than the found point, the nearest last to descend there first
        SubTask ch[4];
        int numCh = 0;
        for ( int c = 0; c < node.count; ++c )
        {
            const int n = node.first + c;
            const float d = boxDistSq( n );
            if ( !( d < res.distSq ) )
                continue;
            int k = numCh++;
            for ( ; k > 0 && ch[k - 1].distSq < d; --k )
                ch[k] = ch[k - 1];
            ch[k] = { n, d };
        }
        for ( int i = 0; i < numCh; ++i )
            subtasks.push( ch[i] );
    }
    resVec[index] = res;
}

void meshProjectionKernel( const float3* points, 
                           const WideNode3* nodes, const float3* meshPoints, const LeafTri3* leafTris,
                           MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = int( ( size + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    kernel<<< numBlocks, maxThreadsPerBlock >>>( points, nodes, meshPoints, leafTris, resVec, xf, refXf, upDistLimitSq, loDistLimitSq, size );
}

}}

