#include "MRCudaPointsToMeshProjector.cuh"
#include "MRCudaInplaceStack.cuh"

namespace MR { namespace Cuda {

__global__ void kernel( const float3* points,
    MeshTree3 tree, const float3* __restrict__ meshPoints,
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
    const LeafTri3* resLeaf = nullptr; // the found triangle
    struct SubTask
    {
        int n;
        float distSq;
    };

    // every level adds at most 3 tasks, and the tree of 2^31 triangles has 16 levels
    InplaceStack<SubTask, 64> subtasks;

    auto boxDistSq = [&] ( const Box3& b )
    {
        const auto box = refXf.isIdentity ? b : refXf.transform( b );
        return lengthSq( box.getBoxClosestPointTo( pt ) - pt );
    };

    // the node to visit next is not pushed in the stack
    SubTask s{ tree.rootRef, boxDistSq( tree.rootBox ) };
    while ( true )
    {
        if ( s.distSq < res.distSq )
        {
            if ( s.n >= 0 )
            {
                const SimdNode3 node = tree.nodes[s.n]; // to registers
                // the children closer than the found point ordered by distance, the farthest first
                SubTask ch[4];
                int numCh = 0;
                #pragma unroll
                for ( int c = 0; c < 4 && node.child[c]; ++c )
                {
                    const int n = node.child[c];
                    const float d = boxDistSq( node.childBox( c ) );
                    if ( !( d < res.distSq ) )
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
                bool stop = false;
                const auto* lts = tree.leafSlots( s.n );
                for ( int i = 0; i < cMaxLeafTris && lts[i].face >= 0; ++i )
                {
                    const auto& lt = lts[i];
                    const float3 a = meshPoints[lt.verts[0]];
                    const float3 b = meshPoints[lt.verts[1]];
                    const float3 c = meshPoints[lt.verts[2]];

                    ClosestPointRes closestPointRes;
                    float distSq;
                    if ( refXf.isIdentity )
                    {
                        closestPointRes = closestPointInTriangle( pt, a, b, c );
                        distSq = lengthSq( closestPointRes.proj - pt );
                    }
                    else
                    {
                        // transform only the first vertex as a point and the edges as vectors, and search relative to the first vertex:
                        // transforming all three vertices in float distorts the shape of small triangles far from the origin,
                        // and the projections of far points on them become very inaccurate
                        const float3 ta = refXf.transform( a );
                        closestPointRes = closestPointInTriangle( pt - ta, float3{ 0, 0, 0 }, refXf.transformVector( b - a ), refXf.transformVector( c - a ) );
                        distSq = lengthSq( closestPointRes.proj - ( pt - ta ) );
                        closestPointRes.proj = closestPointRes.proj + ta;
                    }
                    if ( distSq < res.distSq )
                    {
                        res.distSq = distSq;
                        res.proj.point = closestPointRes.proj;
                        res.proj.faceId = lt.face;
                        resLeaf = &lt;
                        res.tp = TriPoint{ closestPointRes.bary.x, closestPointRes.bary.y };
                        if ( distSq <= loDistLimitSq )
                        {
                            stop = true;
                            break;
                        }
                    }
                }
                if ( stop )
                    break;
            }
        }
        if ( subtasks.empty() )
            break;
        s = subtasks.top();
        subtasks.pop();
    }
    if ( !refXf.isIdentity && resLeaf )
    {
        // return the point in the space of the reference mesh as in case of rigid refXf (which is applied to the points instead)
        const auto& lt = *resLeaf;
        const float3 a = meshPoints[lt.verts[0]];
        res.proj.point = a + ( meshPoints[lt.verts[1]] - a ) * res.tp.a + ( meshPoints[lt.verts[2]] - a ) * res.tp.b;
    }
    resVec[index] = res;
}

void meshProjectionKernel( const float3* points, 
                           MeshTree3 tree, const float3* meshPoints,
                           MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = int( ( size + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    kernel<<< numBlocks, maxThreadsPerBlock >>>( points, tree, meshPoints, resVec, xf, refXf, upDistLimitSq, loDistLimitSq, size );
}

}}

