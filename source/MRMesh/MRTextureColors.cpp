#include "MRTextureColors.h"
#include "MRMesh.h"
#include "MRTriMesh.h"
#include "MRMeshTexture.h"
#include "MRColor.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRVector2.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRPch/MRTBB.h"

#include <atomic>
#include <cstdint>
#include <limits>

namespace MR
{

static UVCoord getUV( VertId v, const ThreeVertIds& tri, const ThreeUVCoords& uvs )
{
    for ( int i = 0; i < 3; ++i )
    {
        if ( tri[i] != v )
            continue;
        return uvs[i];
    }
    assert( false ); // no such vertex in the triangle
    return {};
}

std::optional<UVCoord> findVertexUV( const MeshTopology& topology, VertId v, const TriCornerUVCoords& triCornerUvCoords )
{
    std::optional<UVCoord> res;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        auto l = topology.left( e );
        if ( !l )
            continue;
        const auto uv = getUV( v, topology.getTriVerts( l ), triCornerUvCoords[l] );
        if ( !res )
        {
            res = uv;
            continue;
        }
        if ( *res != uv )
        {
            res = std::nullopt;
            return res;
        }
        // *res == uv, continue
    }
    return res;
}

std::optional<VertUVCoords> findVertexUVs( const MeshTopology& topology, const TriCornerUVCoords& triCornerUvCoords )
{
    MR_TIMER;
    VertUVCoords res;
    res.resizeNoInit( topology.vertSize() );
    tbb::task_group_context ctx;
    ParallelFor( 0_v, VertId( topology.vertSize() ), [&]( VertId v )
    {
        auto maybeUV = findVertexUV( topology, v, triCornerUvCoords );
        if ( maybeUV )
            res[v] = *maybeUV;
        else
            ctx.cancel_group_execution();
    }, ctx );

    if ( ctx.is_group_execution_cancelled() )
        return std::nullopt;
    return res;
}

std::optional<VertUVCoords> findVertexUVs( const Triangulation& tris, const TriCornerUVCoords& triCornerUvCoords )
{
    MR_TIMER;
    assert( triCornerUvCoords.size() >= tris.size() );
    const int maxVert = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, tris.size() ), -1,
        [&] ( const tbb::blocked_range<size_t>& range, int m )
        {
            for ( size_t i = range.begin(); i < range.end(); ++i )
                for ( auto v : tris[FaceId( i )] )
                    m = std::max( m, int( v ) );
            return m;
        },
        [] ( int a, int b ) { return std::max( a, b ); } );

    // 1 + some triangle of each vertex, 0 for the vertex without triangles
    std::vector<std::atomic<int>> vertTri( maxVert + 1 );
    ParallelFor( tris, [&] ( FaceId f )
    {
        for ( auto v : tris[f] )
            if ( v )
                vertTri[v].store( int( f ) + 1, std::memory_order_relaxed );
    } );

    VertUVCoords res;
    res.resizeNoInit( vertTri.size() );
    tbb::task_group_context ctx;
    ParallelFor( 0_v, res.endId(), [&] ( VertId v )
    {
        const int t = vertTri[v].load( std::memory_order_relaxed );
        if ( t == 0 )
        {
            ctx.cancel_group_execution();
            return;
        }
        const FaceId f( t - 1 );
        res[v] = getUV( v, tris[f], triCornerUvCoords[f] );
    }, ctx );
    if ( ctx.is_group_execution_cancelled() )
        return std::nullopt;

    // all triangles of each vertex must have the same uv-coordinates in it
    ParallelFor( tris, [&] ( FaceId f )
    {
        for ( int i = 0; i < 3; ++i )
            if ( const auto v = tris[f][i]; v && triCornerUvCoords[f][i] != res[v] )
                ctx.cancel_group_execution();
    }, ctx );
    if ( ctx.is_group_execution_cancelled() )
        return std::nullopt;
    return res;
}

Color sampleVertexColor( const Mesh& mesh, VertId v, const MeshTexture& tex, const TriCornerUVCoords& triCornerUvCoords )
{
    Vector4f sumWC;
    float sumW = 0;
    for ( EdgeId e : orgRing( mesh.topology, v ) )
    {
        auto l = mesh.topology.left( e );
        if ( !l )
            continue;
        //auto lvs = mesh.topology.getTriVerts( l );
        const auto d0 = edgeVector( mesh.topology, mesh.points, e );
        const auto d1 = edgeVector( mesh.topology, mesh.points, mesh.topology.next( e ) );
        const auto angle = MR::angle( d0, d1 );
        sumWC += angle * Vector4f( tex.sample( tex.filter, getUV( v, mesh.topology.getTriVerts( l ), triCornerUvCoords[l] ) ) );
        sumW += angle;
    }
    if ( sumW > 0 )
        return Color( sumWC/ sumW );
    return {};
}

VertColors sampleVertexColors( const Mesh& mesh, const MeshTexture& tex, const TriCornerUVCoords& triCornerUvCoords )
{
    MR_TIMER;
    VertColors res;
    res.resizeNoInit( mesh.points.size() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        res[v] = sampleVertexColor( mesh, v, tex, triCornerUvCoords );
    } );
    return res;
}

VertColors sampleVertexColors( const TriMesh& mesh, const MeshTexture& tex, const TriCornerUVCoords& triCornerUvCoords )
{
    MR_TIMER;
    assert( triCornerUvCoords.size() >= mesh.tris.size() );
    assert( 3 * mesh.tris.size() <= std::numeric_limits<std::uint32_t>::max() );
    const auto numVerts = mesh.points.size();

    // triangle corners (3 * triangle + index in triangle) of each vertex, grouped by vertices
    std::vector<std::atomic<std::uint32_t>> nextCorner( numVerts );
    ParallelFor( mesh.tris, [&] ( FaceId f )
    {
        for ( auto v : mesh.tris[f] )
            nextCorner[v].fetch_add( 1, std::memory_order_relaxed );
    } );
    std::vector<std::uint32_t> firstCorner( numVerts + 1 );
    for ( size_t i = 0; i < numVerts; ++i )
    {
        const auto n = nextCorner[i].load( std::memory_order_relaxed );
        firstCorner[i + 1] = firstCorner[i] + n;
        nextCorner[i].store( firstCorner[i], std::memory_order_relaxed );
    }
    std::vector<std::uint32_t> corners( firstCorner.back() );
    ParallelFor( mesh.tris, [&] ( FaceId f )
    {
        for ( int i = 0; i < 3; ++i )
            corners[nextCorner[mesh.tris[f][i]].fetch_add( 1, std::memory_order_relaxed )] = std::uint32_t( 3 * size_t( f ) + i );
    } );

    VertColors res;
    res.resizeNoInit( numVerts );
    ParallelFor( res, [&] ( VertId v )
    {
        const auto vCorners = corners.begin() + firstCorner[v];
        const auto vCornersEnd = corners.begin() + firstCorner[size_t( v ) + 1];
        std::sort( vCorners, vCornersEnd ); // the same summation order in every run
        Vector4f sumWC;
        float sumW = 0;
        for ( auto it = vCorners; it != vCornersEnd; ++it )
        {
            const FaceId f( int( *it / 3 ) );
            const int i = int( *it % 3 );
            const auto& vs = mesh.tris[f];
            const auto& p = mesh.points[v];
            const auto angle = MR::angle( mesh.points[vs[( i + 1 ) % 3]] - p, mesh.points[vs[( i + 2 ) % 3]] - p );
            sumWC += angle * Vector4f( tex.sample( tex.filter, triCornerUvCoords[f][i] ) );
            sumW += angle;
        }
        res[v] = sumW > 0 ? Color( sumWC / sumW ) : Color{};
    } );
    return res;
}

void convertTriCornerUVs( const Mesh& mesh, TriCornerUVCoords& triCornerUvCoords, VertUVCoords& uvCoords, VertColors& colors, MeshTexture& texture )
{
    if ( triCornerUvCoords.empty() || mesh.topology.numValidFaces() <= 0 )
        return;

    if ( auto vertUVs = findVertexUVs( mesh.topology, triCornerUvCoords ) )
    {
        uvCoords = std::move( *vertUVs );
        triCornerUvCoords.clear();
        return;
    }

    if ( texture.pixels.empty() )
        return;
    colors = sampleVertexColors( mesh, texture, triCornerUvCoords );
    if ( uvCoords.empty() )
        texture = {}; // texture will not be used without uv-coordinates
}

} //namespace MR
