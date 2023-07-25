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

FlowAggregator::FlowAggregator( const Mesh & mesh, const VertScalars & field ) : mesh_( mesh ), field_( field )
{
    MR_TIMER
    downFlowVert_.resize( mesh.topology.vertSize() );
    downPath_.resize( mesh.topology.vertSize() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        const EdgePoint p0( mesh.topology, v );
        VertId nextVert;
        downPath_[v] = computeSteepestDescentPath( mesh, field, p0, {}, &nextVert );
        downFlowVert_[v] = nextVert;
    } );

    using MinusHeightVert = std::pair<float, VertId>;
    std::vector<MinusHeightVert> minusHeightVerts;
    minusHeightVerts.reserve( mesh.topology.numValidVerts() );
    for ( auto v : mesh.topology.getValidVerts() )
        minusHeightVerts.push_back( { -field[v], v } );
    tbb::parallel_sort( minusHeightVerts.begin(), minusHeightVerts.end() );

    vertsSortedDesc_.reserve( minusHeightVerts.size() );
    for ( size_t i = 0; i < minusHeightVerts.size(); ++i )
        vertsSortedDesc_.push_back( minusHeightVerts[i].second );
}

VertScalars FlowAggregator::computeFlow( const std::vector<FlowOrigin> & starts, Polyline3 * outPolyline, UndirectedEdgeScalars * outFlowPerEdge )
{
    MR_TIMER
    assert( !outFlowPerEdge || outPolyline );

    VertScalars flowInVert( mesh_.topology.vertSize() );
    std::vector<VertId> start2downVert( starts.size() ); // for each start point stores what next vertex is on flow path (can be invalid)
    std::vector<SurfacePath> start2downPath( starts.size() ); // till next vertex

    ParallelFor( starts, [&]( size_t i )
    {
        VertId nextVert;
        start2downPath[i] = computeSteepestDescentPath( mesh_, field_, starts[i].point, {}, &nextVert );
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

    for ( size_t i = 0; i < starts.size(); ++i )
    {
        MeshTriPoint end;
        if ( auto v = start2downVert[i] )
        {
            flowInVert[v] += starts[i].amount;
            end = MeshTriPoint{ mesh_.topology, v };
        }
        addPath( starts[i].amount, starts[i].point, start2downPath[i], end );
    }

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
