#include "MRExtractIsolines.h"
#include "MRBitSet.h"
#include "MREdgeIterator.h"
#include "MREdgePoint.h"
#include "MRPlane3.h"
#include "MRMesh.h"
#include "MRAffineXf3.h"
#include "MRVector2.h"
#include "MRRingIterator.h"
#include "MRRegionBoundary.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

using ValueInVertex = std::function<float( VertId )>;
using ContinueTrack = std::function<bool( const MeshEdgePoint& )>;

class Isoliner
{
public:
    Isoliner( const MeshTopology& topology, ValueInVertex valueInVertex, const FaceBitSet* region )
        : topology_( topology ), region_( region ), valueInVertex_( valueInVertex )
        { findNegativeVerts_(); }

    bool hasAnyLine() const;
    IsoLines extract();

    IsoLine track( const MeshTriPoint& start, ContinueTrack continueTrack );

private:
    void findNegativeVerts_();
    // if continueTrack is not set extract all
    // if continueTrack is set - extract until reach it or closed, or border faced
    IsoLine extractOneLine_( EdgeId first, ContinueTrack continueTrack = {} );
    MeshEdgePoint toEdgePoint_( EdgeId e ) const;
    void computePointOnEachEdge_( IsoLine & line );
    EdgeId findNextEdge_( EdgeId e ) const;

private:
    const MeshTopology& topology_;
    const FaceBitSet* region_ = nullptr;
    ValueInVertex valueInVertex_;
    VertBitSet negativeVerts_;
    UndirectedEdgeBitSet seenEdges_;
};

void Isoliner::findNegativeVerts_()
{
    VertBitSet store;
    const auto & vertRegion = getIncidentVerts( topology_, region_, store );
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
    std::vector<std::vector<MeshEdgePoint>> res;
    for ( auto ue : undirectedEdges( topology_ ) )
    {
        if ( region_ && !contains( *region_, topology_.left( ue ) ) && !contains( *region_, topology_.right( ue ) ) )
            continue;
        if ( seenEdges_.test( ue ) )
            continue;
        EdgeId e = ue;
        VertId o = topology_.org( e );
        VertId d = topology_.dest( e );
        auto no = negativeVerts_.test( o );
        auto nd = negativeVerts_.test( d );
        if ( no == nd )
            continue;

        res.push_back( extractOneLine_( no ? e : e.sym() ) );
    }
    return res;
}

bool Isoliner::hasAnyLine() const
{
    for ( auto ue : undirectedEdges( topology_ ) )
    {
        if ( region_ && !contains( *region_, topology_.left( ue ) ) && !contains( *region_, topology_.right( ue ) ) )
            continue;
        EdgeId e = ue;
        VertId o = topology_.org( e );
        VertId d = topology_.dest( e );
        auto no = negativeVerts_.test( o );
        auto nd = negativeVerts_.test( d );
        if ( no != nd )
            return true;
    }
    return false;
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
        startEdge = testEdge( eOp->e );
        if ( !startEdge )
            startEdge = eOp->e.sym();
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

void Isoliner::computePointOnEachEdge_( IsoLine & line )
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
    std::vector<MeshEdgePoint> res;
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
    seenEdges_.autoResizeSet( first.undirected() );

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
        seenEdges_.autoResizeSet( next.undirected() );
    }

    if ( continueTrack )
        return res;

    if ( !closed )
    {
        auto firstSym = first;
        firstSym = firstSym.sym(); // go backward
        std::vector<MeshEdgePoint> back;
        back.push_back( MeshEdgePoint( firstSym, -1 ) );
        while ( auto next = findNextEdge_( back.back().e ) )
        {
            back.push_back( MeshEdgePoint( next, -1 ) );
            seenEdges_.autoResizeSet( next.undirected() );
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
    const Vector<float, VertId>& vertValues, float isoValue, const FaceBitSet* region )
{
    MR_TIMER;

    Isoliner s( topology, [&] ( VertId v )
    {
        return vertValues[v] - isoValue;
    }, region );
    return s.extract();
}

bool hasAnyIsoline( const MeshTopology& topology,
    const Vector<float, VertId>& vertValues, float isoValue, const FaceBitSet* region )
{
    MR_TIMER;

    Isoliner s( topology, [&] ( VertId v )
    {
        return vertValues[v] - isoValue;
    }, region );
    return s.hasAnyLine();
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
    ValueInVertex valueInVertex = [&] ( VertId v )
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
