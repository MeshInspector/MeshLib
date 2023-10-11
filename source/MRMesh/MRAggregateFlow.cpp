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
        if ( mesh.topology.isBdVertex( v ) )
            return;
        const EdgePoint p0( mesh.topology, v );
        VertId nextVert;
        EdgePoint bdPoint;
        downPath_[v] = computeSteepestDescentPath( mesh, heights, p0, { .outVertexReached = &nextVert, .outBdReached = &bdPoint } );
        if ( bdPoint )
            downPath_[v].push_back( bdPoint );
        downFlowVert_[v] = nextVert;
    } );

    rootVert_.resize( mesh.topology.vertSize() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        auto root = v;
        while ( auto n = downFlowVert_[root] )
            root = n;
        rootVert_[v] = root;
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

VertScalars FlowAggregator::computeFlow( const std::vector<FlowOrigin> & starts, const OutputFlows & out ) const
{
    return computeFlow( starts.size(),
        [&starts]( size_t n ) { return starts[n].point; },
        [&starts]( size_t n ) { return starts[n].amount; },
        {}, out );
}

VertScalars FlowAggregator::computeFlow( const std::vector<MeshTriPoint> & starts, const OutputFlows & out ) const
{
    return computeFlow( starts.size(),
        [&starts]( size_t n ) { return starts[n]; },
        []( size_t ) { return 1.0f; },
        {}, out );
}

VertScalars FlowAggregator::computeFlow( size_t numStarts,
    const std::function<MeshTriPoint(size_t)> & startById,
    const std::function<float(size_t)> & amountById,
    const std::function<const FaceBitSet*(size_t)> & regionById,
    const OutputFlows & out ) const
{
    MR_TIMER
    assert( !out.pFlowPerEdge || out.pPolyline );

    VertScalars flowInVert( mesh_.topology.vertSize() );
    std::vector<VertId> start2downVert( numStarts ); // for each start point stores what next vertex is on flow path (can be invalid)
    std::vector<SurfacePath> start2downPath( numStarts ); // till next vertex

    ParallelFor( start2downVert, [&]( size_t i )
    {
        VertId nextVert;
        if ( auto s = startById( i ); s && !s.isBd( mesh_.topology ) )
        {
            MeshPart mp{ mesh_ };
            if ( regionById )
                mp.region = regionById( i );
            EdgePoint bdPoint;
            start2downPath[i] = computeSteepestDescentPath( mp, heights_, s, { .outVertexReached = &nextVert, .outBdReached = &bdPoint } );
            if ( bdPoint )
                start2downPath[i].push_back( bdPoint );
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

    if ( out.pPolyline )
    {
        std::vector<VertId> sample2firstPolylineVert;
        sample2firstPolylineVert.reserve( numStarts + vertsSortedDesc_.size() + 1 );
        VertId n = 0_v;
        sample2firstPolylineVert.push_back( n );
        // lines from sample starts to first mesh vertices
        for ( size_t i = 0; i < numStarts; ++i )
        {
            if ( amountById( i ) > out.amountGreaterThan && ( !start2downPath[i].empty() || start2downVert[i] ) )
                n += 1 + (int)start2downPath[i].size() + start2downVert[i].valid();
            sample2firstPolylineVert.push_back( n );
        };
        // lines from vertices to down vertices
        for ( size_t i = 0; i < vertsSortedDesc_.size(); ++i )
        {
            auto vUp = vertsSortedDesc_[i];
            if ( flowInVert[vUp] > out.amountGreaterThan && ( !downPath_[vUp].empty() || downFlowVert_[vUp] ) )
                n += 1 + (int)downPath_[vUp].size() + downFlowVert_[vUp].valid();
            sample2firstPolylineVert.push_back( n );
        }

        VertCoords points;
        points.resizeNoInit( n );
        if ( out.pFlowPerEdge )
            out.pFlowPerEdge->resize( n );
        ParallelFor( start2downVert, [&]( size_t i )
        {
            VertId j = sample2firstPolylineVert[i];
            const VertId jEnd = sample2firstPolylineVert[i+1];
            if ( j == jEnd )
                return;
            if ( out.pFlowPerEdge )
            {
                const auto f = amountById( i );
                for ( auto k = j; k < jEnd; ++k )
                    (*out.pFlowPerEdge)[UndirectedEdgeId( (int)k )] = f;
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
            if ( out.pFlowPerEdge )
            {
                const auto f = flowInVert[vUp];
                for ( auto k = j; k < jEnd; ++k )
                    (*out.pFlowPerEdge)[UndirectedEdgeId( (int)k )] = f;
            }
            points[j++] = mesh_.points[vUp];
            for ( const auto & ep : downPath_[vUp] )
                points[j++] = mesh_.edgePoint( ep );
            if ( auto v = downFlowVert_[vUp] )
                points[j++] = mesh_.points[v];
            assert( j == jEnd );
        } );
        *out.pPolyline = Polyline3( sample2firstPolylineVert, points );
    }

    return flowInVert;
}

auto FlowAggregator::computeFlowsPerBasin( size_t numStarts,
    const std::function<MeshTriPoint(size_t)> & startById,
    const std::function<float(size_t)> & amountById ) const -> HashMap<VertId, Flows>
{
    MR_TIMER

    VertScalars flowInVert( mesh_.topology.vertSize() );
    std::vector<VertId> start2downVert( numStarts ); // for each start point stores what next vertex is on flow path (can be invalid)
    std::vector<VertId> start2rootVert( numStarts );
    std::vector<SurfacePath> start2downPath( numStarts ); // till next vertex

    ParallelFor( start2downVert, [&]( size_t i )
    {
        VertId nextVert;
        if ( auto s = startById( i ); s && !s.isBd( mesh_.topology ) )
        {
            EdgePoint bdPoint;
            start2downPath[i] = computeSteepestDescentPath( mesh_, heights_, s, { .outVertexReached = &nextVert, .outBdReached = &bdPoint } );
            if ( bdPoint )
                start2downPath[i].push_back( bdPoint );
            start2downVert[i] = nextVert;
            if ( nextVert )
                start2rootVert[i] = rootVert_[nextVert];
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

    HashMap<VertId, std::vector<VertId>> root2firstPolylineVert;

    // paths from sample starts to first mesh vertices
    std::vector<size_t> start2numInRoot;
    start2numInRoot.reserve( numStarts );
    for ( size_t i = 0; i < numStarts; ++i )
    {
        const auto r = start2rootVert[i];
        auto & f = root2firstPolylineVert[r];
        VertId n;
        if ( f.empty() )
            f.push_back( n = 0_v );
        else
            n = f.back();
        start2numInRoot.push_back( f.size() - 1 );
        if ( !start2downPath[i].empty() || start2downVert[i] )
            n += 1 + (int)start2downPath[i].size() + start2downVert[i].valid();
        f.push_back( n );
    };

    // paths from mesh vertices
    Vector<size_t, VertId> vert2numInRoot;
    vert2numInRoot.resize( rootVert_.size() );
    for ( size_t i = 0; i < vertsSortedDesc_.size(); ++i )
    {
        auto vUp = vertsSortedDesc_[i];
        const auto r = rootVert_[vUp];
        auto & f = root2firstPolylineVert[r];
        VertId n;
        if ( f.empty() )
            f.push_back( n = 0_v );
        else
            n = f.back();
        vert2numInRoot[vUp] = f.size() - 1;
        if ( flowInVert[vUp] && ( !downPath_[vUp].empty() || downFlowVert_[vUp] ) )
            n += 1 + (int)downPath_[vUp].size() + downFlowVert_[vUp].valid();
        f.push_back( n );
    }

    HashMap<VertId, Flows> res;
    for ( const auto & [r,v] : root2firstPolylineVert )
    {
        auto & x = res[r];
        assert( !v.empty() );
        const auto n = v.back();
        x.polyline.points.resizeNoInit( n );
        x.flowPerEdge.resize( n );
    }
        
    // paths from sample starts to first mesh vertices
    ParallelFor( start2downVert, [&]( size_t i )
    {
        const auto r = start2rootVert[i];
        const auto it = root2firstPolylineVert.find( r );
        assert( it != root2firstPolylineVert.end() );
        const auto & f = it->second;
        const auto l = start2numInRoot[i];
        VertId j = f[l];
        const VertId jEnd = f[l+1];
        if ( j == jEnd )
            return;
        auto & x = res[r];
        const auto a = amountById( i );
        for ( auto k = j; k < jEnd; ++k )
            x.flowPerEdge[UndirectedEdgeId( (int)k )] = a;
        x.polyline.points[j++] = mesh_.triPoint( startById( i ) );
        for ( const auto & ep : start2downPath[i] )
            x.polyline.points[j++] = mesh_.edgePoint( ep );
        if ( auto v = start2downVert[i] )
            x.polyline.points[j++] = mesh_.points[v];
        assert( j == jEnd );
    } );

    // paths from mesh vertices
    ParallelFor( vertsSortedDesc_, [&]( size_t i )
    {
        auto vUp = vertsSortedDesc_[i];
        const auto r = rootVert_[vUp];
        const auto it = root2firstPolylineVert.find( r );
        assert( it != root2firstPolylineVert.end() );
        const auto & f = it->second;
        const auto l = vert2numInRoot[vUp];
        VertId j = f[l];
        const VertId jEnd = f[l+1];
        if ( j == jEnd )
            return;
        auto & x = res[r];
        const auto a = flowInVert[vUp];
        for ( auto k = j; k < jEnd; ++k )
            x.flowPerEdge[UndirectedEdgeId( (int)k )] = a;
        x.polyline.points[j++] = mesh_.points[vUp];
        for ( const auto & ep : downPath_[vUp] )
            x.polyline.points[j++] = mesh_.edgePoint( ep );
        if ( auto v = downFlowVert_[vUp] )
            x.polyline.points[j++] = mesh_.points[v];
        assert( j == jEnd );
    } );

    // make polyline topology
    for ( const auto & [r,v] : root2firstPolylineVert )
    {
        auto & x = res[r];
        assert( !v.empty() );
        x.polyline.topology.buildOpenLines( v );
    }

    return res;
}

auto FlowAggregator::computeFlowsPerBasin( const std::vector<FlowOrigin> & starts ) const -> HashMap<VertId, Flows>
{
    return computeFlowsPerBasin( starts.size(),
        [&starts]( size_t n ) { return starts[n].point; },
        [&starts]( size_t n ) { return starts[n].amount; } );
}

auto FlowAggregator::computeFlowsPerBasin( const std::vector<MeshTriPoint> & starts ) const -> HashMap<VertId, Flows>
{
    return computeFlowsPerBasin( starts.size(),
        [&starts]( size_t n ) { return starts[n]; },
        []( size_t ) { return 1.0f; } );
}

UndirectedEdgeBitSet FlowAggregator::computeCatchmentDelineation() const
{
    MR_TIMER
    Vector<VertId, FaceId> face2rootVert( mesh_.topology.faceSize() );
    BitSetParallelFor( mesh_.topology.getValidFaces(), [&]( FaceId f )
    {
        constexpr auto oneThird = 1.0f / 3;
        const MeshTriPoint start( mesh_.topology.edgeWithLeft( f ), { oneThird, oneThird } );
        VertId nextVert;
        EdgePoint bdPoint;
        computeSteepestDescentPath( mesh_, heights_, start, {}, { .outVertexReached = &nextVert, .outBdReached = &bdPoint } );
        if ( nextVert && !mesh_.topology.isBdVertex( nextVert ) )
            face2rootVert[f] = rootVert_[nextVert];
    } );

    UndirectedEdgeBitSet res( mesh_.topology.undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        auto l = mesh_.topology.left( ue );
        if ( !l )
            return;
        auto r = mesh_.topology.right( ue );
        if ( !r )
            return;
        if ( face2rootVert[l] != face2rootVert[r] )
            res.set( ue );
    } );
    return res;
}

} //namespace MR
