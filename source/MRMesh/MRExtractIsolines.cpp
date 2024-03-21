#include "MRExtractIsolines.h"
#include "MRBitSet.h"
#include "MREdgeIterator.h"
#include "MREdgePoint.h"
#include "MRPlane3.h"
#include "MRMesh.h"
#include "MRMeshIntersect.h"
#include "MRAffineXf3.h"
#include "MRVector2.h"
#include "MRRingIterator.h"
#include "MRRegionBoundary.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include <atomic>

namespace MR
{

using ContinueTrack = std::function<bool( const MeshEdgePoint& )>;

class Isoliner
{
public:
    /// prepares to find iso-lines inside given region (or whole mesh if region==nullptr)
    Isoliner( const MeshTopology& topology, VertMetric valueInVertex, const FaceBitSet* region )
        : topology_( topology ), region_( region ), valueInVertex_( valueInVertex )
        { findNegativeVerts_(); }
    /// prepares to find iso-lines crossing the edges in between given edges
    Isoliner( const MeshTopology& topology, VertMetric valueInVertex, const VertBitSet& vertRegion )
        : topology_( topology ), valueInVertex_( valueInVertex )
        { findNegativeVerts_( vertRegion ); }

    /// if \param potentiallyCrossedEdges is given, then only these edges will be checked (otherwise all mesh edges)
    bool hasAnyLine( const UndirectedEdgeBitSet * potentiallyCrossedEdges = nullptr ) const;

    IsoLines extract();
    /// potentiallyCrossedEdges shall include all edges crossed by the iso-lines (some other edges there is permitted as well)
    IsoLines extract( UndirectedEdgeBitSet potentiallyCrossedEdges );

    IsoLine track( const MeshTriPoint& start, ContinueTrack continueTrack );

private:
    void findNegativeVerts_();
    void findNegativeVerts_( const VertBitSet& vertRegion );
    // if continueTrack is not set extract all
    // if continueTrack is set - extract until reach it or closed, or border faced
    IsoLine extractOneLine_( EdgeId first, ContinueTrack continueTrack = {} );
    MeshEdgePoint toEdgePoint_( EdgeId e ) const;
    void computePointOnEachEdge_( IsoLine & line ) const;
    EdgeId findNextEdge_( EdgeId e ) const;

private:
    const MeshTopology& topology_;
    const FaceBitSet* region_ = nullptr;
    VertMetric valueInVertex_;
    VertBitSet negativeVerts_;
    UndirectedEdgeBitSet activeEdges_; // the edges crossed by the iso-line, but not yet extracted
};

void Isoliner::findNegativeVerts_()
{
    VertBitSet store;
    findNegativeVerts_( getIncidentVerts( topology_, region_, store ) );
}

void Isoliner::findNegativeVerts_( const VertBitSet& vertRegion )
{
    negativeVerts_.clear();
    negativeVerts_.resize( vertRegion.size() );
    BitSetParallelFor( vertRegion, [&]( VertId v )
    {
        if ( valueInVertex_( v ) < 0 )
            negativeVerts_.set( v );
    } );
}

IsoLines Isoliner::extract()
{
    activeEdges_.clear();
    activeEdges_.resize( topology_.undirectedEdgeSize() );
    BitSetParallelForAll( activeEdges_, [&]( UndirectedEdgeId ue )
    {
        VertId o = topology_.org( ue );
        if ( !o )
            return;
        VertId d = topology_.dest( ue );
        if ( !d )
            return;
        auto no = negativeVerts_.test( o );
        auto nd = negativeVerts_.test( d );
        if ( no != nd && ( !region_ || contains( *region_, topology_.left( ue ) ) || contains( *region_, topology_.right( ue ) ) ) )
            activeEdges_.set( ue );
    } );

    IsoLines res;
    for ( auto ue : activeEdges_ )
    {
        EdgeId e = ue;
        VertId o = topology_.org( e );
        auto no = negativeVerts_.test( o );
        assert ( no != negativeVerts_.test( topology_.dest( e ) ) );
        // direct edge from negative to positive values
        res.push_back( extractOneLine_( no ? e : e.sym() ) );
    }
    activeEdges_.clear();
    return res;
}

IsoLines Isoliner::extract( UndirectedEdgeBitSet potentiallyCrossedEdges )
{
    activeEdges_ = std::move( potentiallyCrossedEdges );
    IsoLines res;
    for ( auto ue : activeEdges_ )
    {
        EdgeId e = ue;
        auto no = negativeVerts_.test( topology_.org( e ) );
        auto nd = negativeVerts_.test( topology_.dest( e ) );
        if ( no == nd )
            continue;
        // direct edge from negative to positive values
        res.push_back( extractOneLine_( no ? e : e.sym() ) );
    }
    activeEdges_.clear();
    return res;
}

bool Isoliner::hasAnyLine( const UndirectedEdgeBitSet * potentiallyCrossedEdges ) const
{
    std::atomic<bool> res{ false };
    tbb::parallel_for( tbb::blocked_range( 0_ue, UndirectedEdgeId( topology_.undirectedEdgeSize() ) ),
        [&] ( const tbb::blocked_range<UndirectedEdgeId>& range )
    {
        for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
        {
            if ( res.load( std::memory_order_relaxed ) )
                break;
            if ( potentiallyCrossedEdges && !potentiallyCrossedEdges->test( ue ) )
                continue;
            VertId o = topology_.org( ue );
            if ( !o )
                continue;
            VertId d = topology_.dest( ue );
            if ( !d )
                continue;
            auto no = negativeVerts_.test( o );
            auto nd = negativeVerts_.test( d );
            if ( no != nd )
            {
                assert ( !region_ || contains( *region_, topology_.left( ue ) ) || contains( *region_, topology_.right( ue ) ) );
                res = true;
                break;
            }
        }
    } );
    return res;
}

IsoLine Isoliner::track( const MeshTriPoint& start, ContinueTrack continueTrack )
{
    auto testEdge = [&] ( EdgeId e ) -> EdgeId
    {
        VertId o = topology_.org( e );
        VertId d = topology_.dest( e );
        auto no = negativeVerts_.test( o );
        auto nd = negativeVerts_.test( d );
        return ( nd && !no ) ? e.sym() : EdgeId{};
    };

    EdgeId startEdge;
    if ( auto v = start.inVertex( topology_ ) )
    {
        for ( auto e : orgRing( topology_, v ) )
        {
            if ( auto se = testEdge( e ) )
            {
                startEdge = se;
                break;
            }
        }
    }
    else if ( auto eOp = start.onEdge( topology_ ) )
    {
        startEdge = testEdge( eOp.e );
        if ( !startEdge )
            startEdge = eOp.e.sym();
        startEdge = findNextEdge_( startEdge ); // `start` is first
    }
    else
    {
        for ( auto e : leftRing( topology_, start.e ) )
        {
            if ( auto se = testEdge( e ) )
            {
                startEdge = se;
                break;
            }
        }
    }
    if ( !startEdge )
        return {};
    return extractOneLine_( startEdge, continueTrack );
}

inline MeshEdgePoint Isoliner::toEdgePoint_( EdgeId e ) const
{
    float vo = valueInVertex_( topology_.org( e ) );
    float vd = valueInVertex_( topology_.dest( e ) );
    assert( ( vo < 0 && 0 <= vd ) || ( vd < 0 && 0 <= vo ) );
    const float x = vo / ( vo - vd );
    return MeshEdgePoint( e, x );
}

void Isoliner::computePointOnEachEdge_( IsoLine & line ) const
{
    ParallelFor( line, [&]( size_t i )
    {
        line[i] = toEdgePoint_( line[i].e );
    } );
}

EdgeId Isoliner::findNextEdge_( EdgeId e ) const
{
    if ( !topology_.isLeftInRegion( e, region_ ) )
        return {};
    VertId o, d, x;
    topology_.getLeftTriVerts( e, o, d, x );
    auto no = negativeVerts_.test( o );
    auto nd = negativeVerts_.test( d );
    assert( no != nd );
    auto nx = negativeVerts_.test( x );

    if ( ( no && nx ) || ( nd && !nx ) )
        return topology_.prev( e.sym() ).sym();
    else
        return topology_.next( e );
}

IsoLine Isoliner::extractOneLine_( EdgeId first, ContinueTrack continueTrack )
{
    assert( activeEdges_.empty() || activeEdges_.test( first.undirected() ) );
    IsoLine res;
    auto addCrossedEdge = [&]( EdgeId e )
    {
        if ( !continueTrack )
        {
            // exact point will be found in computePointOnEachEdge_ at the end
            res.push_back( MeshEdgePoint( e, -1 ) );
            return true;
        }
        res.push_back( toEdgePoint_( e ) );
        return continueTrack( res.back() );
    };

    if ( !addCrossedEdge( first ) )
        return res;
    assert( activeEdges_.test( first.undirected() ) );
        activeEdges_.reset( first.undirected() );

    bool closed = false;
    while ( auto next = findNextEdge_( res.back().e ) )
    {
        if ( first == next )
        {
            addCrossedEdge( first );
            closed = true;
            break;
        }
        if ( !addCrossedEdge( next ) )
            return res;
        assert( activeEdges_.empty() || activeEdges_.test( next.undirected() ) );
        activeEdges_.reset( next.undirected() );
    }

    if ( continueTrack )
        return res;

    if ( !closed )
    {
        auto firstSym = first;
        firstSym = firstSym.sym(); // go backward
        IsoLine back;
        back.push_back( MeshEdgePoint( firstSym, -1 ) );
        while ( auto next = findNextEdge_( back.back().e ) )
        {
            back.push_back( MeshEdgePoint( next, -1 ) );
            assert( activeEdges_.empty() || activeEdges_.test( next.undirected() ) );
            activeEdges_.reset( next.undirected() );
        }
        std::reverse( back.begin(), back.end() );
        back.pop_back(); // remove extra copy of firstSym
        for ( auto& i : back )
            i = i.sym(); // make consistent edge orientations of forward and backward passes
        res.insert( res.begin(), back.begin(), back.end() );
    }

    computePointOnEachEdge_( res );
    return res;
}

IsoLines extractIsolines( const MeshTopology& topology,
    const VertMetric & vertValues, const FaceBitSet* region )
{
    MR_TIMER
    Isoliner s( topology, vertValues, region );
    return s.extract();
}

bool hasAnyIsoline( const MeshTopology& topology,
    const VertMetric & vertValues, const FaceBitSet* region )
{
    MR_TIMER
    Isoliner s( topology, vertValues, region );
    return s.hasAnyLine();
}

IsoLines extractIsolines( const MeshTopology & topology,
    const VertScalars & vertValues, float isoValue, const FaceBitSet * region )
{
    return extractIsolines( topology, [&vertValues, isoValue] ( VertId v ) { return vertValues[v] - isoValue; }, region );
}

bool hasAnyIsoline( const MeshTopology & topology,
    const VertScalars & vertValues, float isoValue, const FaceBitSet * region )
{
    return hasAnyIsoline( topology, [&vertValues, isoValue] ( VertId v ) { return vertValues[v] - isoValue; }, region );
}

PlaneSections extractPlaneSections( const MeshPart& mp, const Plane3f& plane )
{
    MR_TIMER;

    Isoliner s( mp.mesh.topology, [&] ( VertId v )
    {
        return plane.distance( mp.mesh.points[v] );
    }, mp.region );
    return s.extract();
}

bool hasAnyPlaneSection( const MeshPart& mp, const Plane3f& plane )
{
    MR_TIMER;

    Isoliner s( mp.mesh.topology, [&] ( VertId v )
    {
        return plane.distance( mp.mesh.points[v] );
    }, mp.region );
    return s.hasAnyLine();
}

PlaneSections extractXYPlaneSections( const MeshPart & mp, float zLevel )
{
    MR_TIMER

    UndirectedEdgeBitSet potentiallyCrossedEdges( mp.mesh.topology.undirectedEdgeSize() );
    VertBitSet vertRegion( mp.mesh.topology.vertSize() );
    xyPlaneMeshIntersect( mp, zLevel, nullptr, &potentiallyCrossedEdges, &vertRegion );

    Isoliner s( mp.mesh.topology, [&points = mp.mesh.points, zLevel] ( VertId v )
    {
        return points[v].z - zLevel;
    }, vertRegion );
    return s.extract( std::move( potentiallyCrossedEdges ) );
}

bool hasAnyXYPlaneSection( const MeshPart & mp, float zLevel )
{
    MR_TIMER

    UndirectedEdgeBitSet potentiallyCrossedEdges( mp.mesh.topology.undirectedEdgeSize() );
    VertBitSet vertRegion( mp.mesh.topology.vertSize() );
    xyPlaneMeshIntersect( mp, zLevel, nullptr, &potentiallyCrossedEdges, &vertRegion );

    Isoliner s( mp.mesh.topology, [&points = mp.mesh.points, zLevel] ( VertId v )
    {
        return points[v].z - zLevel;
    }, vertRegion );
    return s.hasAnyLine( &potentiallyCrossedEdges );
}

PlaneSection trackSection( const MeshPart& mp,
    const MeshTriPoint& start, MeshTriPoint& end, const Vector3f& direction, float distance )
{
    MR_TIMER;
    if ( distance == 0.0f )
    {
        end = start;
        return {};
    }
    const auto dir = distance > 0.0f ? direction : -direction;
    distance = std::abs( distance );
    auto startPoint = mp.mesh.triPoint( start );
    auto prevPoint = startPoint;
    auto plane = Plane3f::fromDirAndPt( cross( dir, mp.mesh.pseudonormal( start ) ), prevPoint );
    VertMetric valueInVertex = [&] ( VertId v )
    {
        return plane.distance( mp.mesh.points[v] );
    };
    ContinueTrack continueTrack = [&] ( const MeshEdgePoint& next ) mutable->bool
    {
        auto point = mp.mesh.edgePoint( next );
        auto dist = ( point - prevPoint ).length();
        distance -= dist;
        if ( distance < 0.0f )
            return false;
        prevPoint = point;
        return true;
    };

    Isoliner s( mp.mesh.topology, valueInVertex, mp.region );
    auto res = s.track( start, continueTrack );
    if ( res.empty() )
    {
        end = start;
        return {};
    }
    bool closed = res.size() != 1 && res.front() == res.back();
    if ( distance > 0.0f )
    {
        end = res.back();
        res.pop_back();
        if ( closed )
            end = start;
        return res;
    }

    auto lastEdgePoint = res.back();
    auto lastPoint = mp.mesh.edgePoint( lastEdgePoint );
    res.pop_back();
    auto lastSegmentLength = ( lastPoint - prevPoint ).length();
    float ratio = ( lastSegmentLength + distance ) / lastSegmentLength;
    auto endPoint = ( 1.0f - ratio ) * prevPoint + ratio * lastPoint;
    end = mp.mesh.toTriPoint( mp.mesh.topology.right( lastEdgePoint.e ), endPoint );
    if ( closed &&
        dot( endPoint - prevPoint, lastPoint - prevPoint ) > dot( startPoint - prevPoint, lastPoint - prevPoint ) )
        end = start;
    return res;
}

Contour2f planeSectionToContour2f( const Mesh& mesh, const PlaneSection& section, const AffineXf3f& meshToPlane )
{
    MR_TIMER;
    Contour2f res;
    res.reserve( section.size() );
    for ( const auto& s : section )
    {
        auto p = meshToPlane( mesh.edgePoint( s ) );
        res.emplace_back( p.x, p.y );
    }

    return res;
}

Contours2f planeSectionsToContours2f( const Mesh& mesh, const PlaneSections& sections, const AffineXf3f& meshToPlane )
{
    MR_TIMER;
    Contours2f res;
    res.reserve( sections.size() );
    for ( const auto& s : sections )
        res.push_back( planeSectionToContour2f( mesh, s, meshToPlane ) );
    return res;
}

} //namespace MR
