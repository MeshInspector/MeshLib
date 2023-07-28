#include "MRAggregateFlow.h"
#include "MRSurfacePath.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRPolyline.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

FlowAggregator::FlowAggregator( const Mesh & mesh, const VertScalars & heights ) : mesh_( mesh ), heights_( heights )
{
    MR_TIMER
    downFlowVert_.resize( mesh.topology.vertSize() );
    downPath_.resize( mesh.topology.vertSize() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        const EdgePoint p0( mesh.topology, v );
        VertId nextVert;
        downPath_[v] = computeSteepestDescentPath( mesh, heights, p0, {}, &nextVert );
        downFlowVert_[v] = nextVert;
    } );

    using MinusHeightVert = std::pair<float, VertId>;
    std::vector<MinusHeightVert> minusHeightVerts;
    minusHeightVerts.reserve( mesh.topology.numValidVerts() );
    for ( auto v : mesh.topology.getValidVerts() )
        minusHeightVerts.push_back( { -heights[v], v } );
    tbb::parallel_sort( minusHeightVerts.begin(), minusHeightVerts.end() );

    vertsSortedDesc_.reserve( minusHeightVerts.size() );
    for ( size_t i = 0; i < minusHeightVerts.size(); ++i )
        vertsSortedDesc_.push_back( minusHeightVerts[i].second );
}

VertScalars FlowAggregator::computeFlow( const std::vector<FlowOrigin> & starts, Polyline3 * outPolyline, UndirectedEdgeScalars * outFlowPerEdge )
{
    MR_TIMER
    assert( !outFlowPerEdge || outPolyline );

    Timer t("1");

    VertScalars flowInVert( mesh_.topology.vertSize() );
    std::vector<VertId> start2downVert( starts.size() ); // for each start point stores what next vertex is on flow path (can be invalid)
    std::vector<SurfacePath> start2downPath( starts.size() ); // till next vertex

    ParallelFor( starts, [&]( size_t i )
    {
        VertId nextVert;
        start2downPath[i] = computeSteepestDescentPath( mesh_, heights_, starts[i].point, {}, &nextVert );
        start2downVert[i] = nextVert;
    } );

    auto addPath = [&]( float flow, const MeshTriPoint & start, const SurfacePath& path, const MeshTriPoint & end )
    {
        if ( !outPolyline || ( path.empty() && !end ) )
            return;
        outPolyline->addFromGeneralSurfacePath( mesh_, start, path, end );
        if ( outFlowPerEdge )
            outFlowPerEdge->resizeWithReserve( outPolyline->topology.undirectedEdgeSize(), flow );
    };

    t.restart( "2" );

    if ( outPolyline )
    {
        VertId n = 0_v;
        std::vector<VertId> sample2firstPolylineVert;
        sample2firstPolylineVert.reserve( starts.size() + 1 );
        sample2firstPolylineVert.push_back( n );
        for ( size_t i = 0; i < starts.size(); ++i )
        {
            if ( !start2downPath[i].empty() || start2downVert[i] )
                n += 1 + (int)start2downPath[i].size() + start2downVert[i].valid();
            sample2firstPolylineVert.push_back( n );
        };
        VertCoords points;
        points.resizeNoInit( n );
        ParallelFor( starts, [&]( size_t i )
        {
            if ( start2downPath[i].empty() && !start2downVert[i] )
                return;
            VertId j = sample2firstPolylineVert[i];
            points[j++] = mesh_.triPoint( starts[i].point );
            for ( const auto & ep : start2downPath[i] )
                points[j++] = mesh_.edgePoint( ep );
            if ( auto v = start2downVert[i] )
                points[j++] = mesh_.points[v];
            assert( j == sample2firstPolylineVert[i+1] );
        } );
        *outPolyline = Polyline3( sample2firstPolylineVert, points );
    }
    for ( size_t i = 0; i < starts.size(); ++i )
    {
        if ( auto v = start2downVert[i] )
            flowInVert[v] += starts[i].amount;
    }

    t.restart( "3" );

    for ( size_t i = 0; i < vertsSortedDesc_.size(); ++i )
    {
        auto vUp = vertsSortedDesc_[i];
        if ( !flowInVert[vUp] )
            continue;
        MeshTriPoint end;
        if ( auto vDn = downFlowVert_[vUp] )
        {
            flowInVert[vDn] += flowInVert[vUp];
            end = MeshTriPoint{ mesh_.topology, vDn };
        }
        addPath( flowInVert[vUp], { mesh_.topology, vUp }, downPath_[vUp], end );
    }

    return flowInVert;
}

} //namespace MR
