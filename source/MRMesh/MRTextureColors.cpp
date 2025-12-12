#include "MRTextureColors.h"
#include "MRMesh.h"
#include "MRMeshTexture.h"
#include "MRColor.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRVector2.h"
#include "MRBitSetParallelFor.h"

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
    tbb::parallel_for( tbb::blocked_range( 0_v, VertId( topology.vertSize() ) ),
        [&] ( const tbb::blocked_range<VertId>& range )
    {
        for ( auto v = range.begin(); v < range.end(); ++v )
        {
            if ( ctx.is_group_execution_cancelled() )
                break;
            auto maybeUV = findVertexUV( topology, v, triCornerUvCoords );
            if ( maybeUV )
            {
                res[v] = *maybeUV;
            }
            else
            {
                ctx.cancel_group_execution();
                break;
            }
        }
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

} //namespace MR
