#include "MRCudaSweptVolume.cuh"

#include "MRCudaLineSegm.cuh"
#include "MRCudaPolyline2Intersect.cuh"
#include "MRCudaPolylineProject.cuh"

namespace MR::Cuda
{

template <typename Func>
__device__ void findEdgesInTool( const Node3* __restrict__ nodes, const float3* __restrict__ points, const int* __restrict__ orgs, const float3 pos, const float toolRadius, const float toolMinHeight, const float toolMaxHeight, Func callback )
{
    const auto toolRadiusSq = toolRadius * toolRadius;

    const Box3 toolBox {
        float3{ -toolRadius, -toolRadius, -toolMaxHeight } + pos,
        float3{ +toolRadius, +toolRadius, -toolMinHeight } + pos,
    };

    InplaceStack<int, 32> subtasks;

    auto addSubTask = [&] ( int n )
    {
        const auto& box = nodes[n].box;
        if ( box.intersection( toolBox ).valid() )
            subtasks.push( n );
    };

    addSubTask( 0 );

    while ( !subtasks.empty() )
    {
        const auto n = subtasks.top();
        subtasks.pop();

        const auto& node = nodes[n];
        if ( node.leaf() )
        {
            const auto ue = node.leafId();
            const auto a = points[orgs[2 * ue + 0]];
            const auto b = points[orgs[2 * ue + 1]];
            const auto proj = closestPointOnLineSegm( pos, a, b );

            const auto dist2Sq = lengthSq( float2{ proj.x - pos.x, proj.y - pos.y } );
            if ( toolRadiusSq < dist2Sq )
                continue;

            const float2 toolPos { sqrt( dist2Sq ), pos.z - proj.z };
            if ( callback( toolPos ) )
                return;
        }
        else
        {
            addSubTask( node.r ); // look at right node later
            addSubTask( node.l ); // look at left node first
        }
    }
}

__device__ float getBoundarySignedDistanceSq( const Box2 box, const float2 p )
{
    const auto distX = max( box.min.x - p.x, p.x - box.max.x );
    const auto distY = max( box.min.y - p.y, p.y - box.max.y );
    if ( distX > 0.f || distY > 0.f )
        return sqr( distX > 0.f ? distX : 0.f ) + sqr( distY > 0.f ? distY : 0.f );
    else
        return -sqr( max( distX, distY ) );
}

__device__ float sqrSgn( float v )
{
    return (float)sgn( v ) * sqr( v );
}

__device__ float sqrtSgn( float v )
{
    return (float)sgn( v ) * sqrt( abs ( v ) );
}

template <typename PosToDistFunc>
__device__ void kernel( float* __restrict__ output, size_t size, VolumeIndexer indexer, Polyline3Data toolpath, float toolRadius, float toolMinHeight, float toolMaxHeight, float padding, PosToDistFunc posToDistFunc )
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= size )
        return;

    const auto point = indexer.toPoint( indexer.toCoord( index ) );

    const auto paddingSq = sqr( padding );

    auto distSgnSq = FLT_MAX;
    const auto getToolDistance = [&] ( const float2 toolPos )
    {
        distSgnSq = min( distSgnSq, posToDistFunc( toolPos ) );
        return distSgnSq < -paddingSq;
    };
    findEdgesInTool( toolpath.nodes, toolpath.points, toolpath.orgs, point, toolRadius, toolMinHeight, toolMaxHeight, getToolDistance );

    output[index] = distSgnSq != FLT_MAX ? sqrtSgn( distSgnSq ) : FLT_MAX;
}

__global__ void kernel( float* __restrict__ output, size_t size, VolumeIndexer indexer, Polyline3Data toolpath, Polyline2Data toolPolyline, float toolRadius, float toolMinHeight, float toolMaxHeight, float padding )
{
    return kernel( output, size, indexer, toolpath, toolRadius, toolMinHeight, toolMaxHeight, padding, [&] ( const float2 toolPos )
    {
        const auto toolDistSq = findProjectionOnPolyline2( toolPos, toolPolyline.nodes, toolPolyline.points, toolPolyline.orgs, sqr( toolPos.x ), 0.f ).distSq;
        const auto toolDistSgn = isPointInsidePolyline( toolPos, toolPolyline.nodes, toolPolyline.points, toolPolyline.orgs ) ? -1.f : +1.f;
        return toolDistSgn * toolDistSq;
    } );
}

__global__ void kernel( float* __restrict__ output, size_t size, VolumeIndexer indexer, Polyline3Data toolpath, FlatEndMillTool tool, float padding )
{
    const Box2 box {
        { -tool.radius, 0.f },
        { +tool.radius, tool.length },
    };
    return kernel( output, size, indexer, toolpath, tool.radius + padding, -padding, tool.length + padding, padding, [&] ( const float2 toolPos )
    {
        return getBoundarySignedDistanceSq( box, toolPos );
    } );
}

__global__ void kernel( float* __restrict__ output, size_t size, VolumeIndexer indexer, Polyline3Data toolpath, BallEndMillTool tool, float padding )
{
    const float2 center { 0.f, tool.radius };
    const Box2 box {
        { -tool.radius, 0.f },
        { +tool.radius, tool.length },
    };
    return kernel( output, size, indexer, toolpath, tool.radius + padding, -padding, tool.length + padding, padding, [&] ( const float2 toolPos )
    {
        if ( toolPos.y <= center.y )
            return sqrSgn( length( toolPos - center ) - tool.radius );
        else
            return getBoundarySignedDistanceSq( box, toolPos );
    } );
}

__global__ void kernel( float* __restrict__ output, size_t size, VolumeIndexer indexer, Polyline3Data toolpath, BullNoseEndMillTool tool, float padding )
{
    const float2 center { tool.radius - tool.cornerRadius, tool.cornerRadius };
    const Box2 box {
        { -tool.radius, 0.f },
        { +tool.radius, tool.length },
    };
    return kernel( output, size, indexer, toolpath, tool.radius + padding, -padding, tool.length + padding, padding, [&] ( const float2 toolPos )
    {
        if ( center.x <= toolPos.x && toolPos.y <= center.y )
            return sqrSgn( length( toolPos - center ) - tool.cornerRadius );
        else
            return getBoundarySignedDistanceSq( box, toolPos );
    } );
}

__global__ void kernel( float* __restrict__ output, size_t size, VolumeIndexer indexer, Polyline3Data toolpath, ChamferEndMillTool tool, float padding )
{
    const float2 a { tool.endRadius, 0.f }, b { tool.radius, tool.cutterHeight };
    const Box2 box {
        { -tool.radius, 0.f },
        { +tool.radius, tool.length },
    };
    return kernel( output, size, indexer, toolpath, tool.radius + padding, -padding, tool.length + padding, padding, [&] ( const float2 toolPos )
    {
        if ( tool.endRadius <= toolPos.x && toolPos.y <= tool.cutterHeight )
        {
            const auto proj = closestPointOnLineSegm( toolPos, a, b );
            const auto ccw = ( cross( b - a, toolPos - a ) > 0.f );
            return lengthSq( proj - toolPos ) * ( ccw ? -1.f : +1.f );
        }
        else
        {
            return getBoundarySignedDistanceSq( box, toolPos );
        }
    } );
}

void computeToolDistanceKernel( float* __restrict__ output, size_t size, VolumeIndexer indexer, Polyline3Data toolpath,
    Polyline2Data toolPolyline, float toolRadius, float toolMinHeight, float toolMaxHeight, float padding )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = int( ( indexer.size() + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    kernel<<< numBlocks, maxThreadsPerBlock >>>( output, size, indexer, toolpath, toolPolyline, toolRadius, toolMinHeight, toolMaxHeight, padding );
}

void computeToolDistanceKernel( float* output, size_t size, VolumeIndexer indexer, Polyline3Data toolpath,
    FlatEndMillTool tool, float padding )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = int( ( indexer.size() + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    kernel<<< numBlocks, maxThreadsPerBlock >>>( output, size, indexer, toolpath, tool, padding );
}

void computeToolDistanceKernel( float* output, size_t size, VolumeIndexer indexer, Polyline3Data toolpath,
    BallEndMillTool tool, float padding )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = int( ( indexer.size() + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    kernel<<< numBlocks, maxThreadsPerBlock >>>( output, size, indexer, toolpath, tool, padding );
}

void computeToolDistanceKernel( float* output, size_t size, VolumeIndexer indexer, Polyline3Data toolpath,
    BullNoseEndMillTool tool, float padding )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = int( ( indexer.size() + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    kernel<<< numBlocks, maxThreadsPerBlock >>>( output, size, indexer, toolpath, tool, padding );
}

void computeToolDistanceKernel( float* output, size_t size, VolumeIndexer indexer, Polyline3Data toolpath,
    ChamferEndMillTool tool, float padding )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = int( ( indexer.size() + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock );
    kernel<<< numBlocks, maxThreadsPerBlock >>>( output, size, indexer, toolpath, tool, padding );
}

} // namespace MR::Cuda
