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
    return computeFlow( starts.size(),
        [&starts]( size_t n ) { return starts[n].point; },
        [&starts]( size_t n ) { return starts[n].amount; },
        outPolyline, outFlowPerEdge );
}

VertScalars FlowAggregator::computeFlow( const std::vector<MeshTriPoint> & starts, Polyline3 * outPolyline, UndirectedEdgeScalars * outFlowPerEdge )
{
    return computeFlow( starts.size(),
        [&starts]( size_t n ) { return starts[n]; },
        []( size_t ) { return 1.0f; },
        outPolyline, outFlowPerEdge );
}

VertScalars FlowAggregator::computeFlow( size_t numStarts,
    const std::function<MeshTriPoint(size_t)> & startById,
    const std::function<float(size_t)> & amountById,
    Polyline3 * outPolyline, UndirectedEdgeScalars * outFlowPerEdge )
{
    MR_TIMER
    assert( !outFlowPerEdge || outPolyline );

    VertScalars flowInVert( mesh_.topology.vertSize() );
    std::vector<VertId> start2downVert( numStarts ); // for each start point stores what next vertex is on flow path (can be invalid)
    std::vector<SurfacePath> start2downPath( numStarts ); // till next vertex

    ParallelFor( start2downVert, [&]( size_t i )
    {
        VertId nextVert;
        if ( auto s = startById( i ) )
        {
            start2downPath[i] = computeSteepestDescentPath( mesh_, heights_, s, {}, &nextVert );
            start2downVert[i] = nextVert;
        }
    } );

    for ( size_t i = 0; i < numStarts; ++i )
    {
        if ( auto v = start2downVert[i] )
            flowInVert[v] += amountById( i );
    }

    for ( size_t i = 0; i < vertsSortedDesc_.size(); ++i )
    {
        auto vUp = vertsSortedDesc_[i];
        if ( !flowInVert[vUp] )
            continue;
        if ( auto vDn = downFlowVert_[vUp] )
            flowInVert[vDn] += flowInVert[vUp];
    }

    if ( outPolyline )
    {
        std::vector<VertId> sample2firstPolylineVert;
        sample2firstPolylineVert.reserve( numStarts + vertsSortedDesc_.size() + 1 );
        VertId n = 0_v;
        sample2firstPolylineVert.push_back( n );
        for ( size_t i = 0; i < numStarts; ++i )
        {
            if ( !start2downPath[i].empty() || start2downVert[i] )
                n += 1 + (int)start2downPath[i].size() + start2downVert[i].valid();
            sample2firstPolylineVert.push_back( n );
        };
        //polyline vertices [0,n) contain points from sample starts to first mesh vertices
        for ( size_t i = 0; i < vertsSortedDesc_.size(); ++i )
        {
            auto vUp = vertsSortedDesc_[i];
            if ( flowInVert[vUp] && ( !downPath_[vUp].empty() || downFlowVert_[vUp] ) )
                n += 1 + (int)downPath_[vUp].size() + downFlowVert_[vUp].valid();
            sample2firstPolylineVert.push_back( n );
        }

        VertCoords points;
        points.resizeNoInit( n );
        if ( outFlowPerEdge )
            outFlowPerEdge->resize( n );
        ParallelFor( start2downVert, [&]( size_t i )
        {
            VertId j = sample2firstPolylineVert[i];
            const VertId jEnd = sample2firstPolylineVert[i+1];
            if ( j == jEnd )
                return;
            if ( outFlowPerEdge )
            {
                const auto f = amountById( i );
                for ( auto k = j; k < jEnd; ++k )
                    (*outFlowPerEdge)[UndirectedEdgeId( (int)k )] = f;
            }
            points[j++] = mesh_.triPoint( startById( i ) );
            for ( const auto & ep : start2downPath[i] )
                points[j++] = mesh_.edgePoint( ep );
            if ( auto v = start2downVert[i] )
                points[j++] = mesh_.points[v];
            assert( j == jEnd );
        } );
        ParallelFor( vertsSortedDesc_, [&]( size_t i )
        {
            VertId j = sample2firstPolylineVert[numStarts + i];
            const VertId jEnd = sample2firstPolylineVert[numStarts + i + 1];
            if ( j == jEnd )
                return;
            const VertId vUp = vertsSortedDesc_[i];
            if ( outFlowPerEdge )
            {
                const auto f = flowInVert[vUp];
                for ( auto k = j; k < jEnd; ++k )
                    (*outFlowPerEdge)[UndirectedEdgeId( (int)k )] = f;
            }
            points[j++] = mesh_.points[vUp];
            for ( const auto & ep : downPath_[vUp] )
                points[j++] = mesh_.edgePoint( ep );
            if ( auto v = downFlowVert_[vUp] )
                points[j++] = mesh_.points[v];
            assert( j == jEnd );
        } );
        *outPolyline = Polyline3( sample2firstPolylineVert, points );
    }

    return flowInVert;
}

} //namespace MR
