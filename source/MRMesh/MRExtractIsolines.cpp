#include "MRExtractIsolines.h"
#include "MRBitSet.h"
#include "MREdgeIterator.h"
#include "MRMeshEdgePoint.h"
#include "MRPlane3.h"
#include "MRMesh.h"
#include "MRAffineXf3.h"
#include "MRVector2.h"
#include "MRRingIterator.h"
#include "MRTimer.h"

namespace MR
{

using ValueInVertex = std::function<float(VertId)>;
using ContinueTrack = std::function<bool( const MeshEdgePoint& )>;

class Isoliner
{
public:
    Isoliner( const MeshTopology & topology, ValueInVertex valueInVertex, const FaceBitSet * region )
        : topology_( topology ), region_( region ), valueInVertex_( valueInVertex ) { }

    IsoLines extract();

    IsoLine track( const MeshTriPoint& start, ContinueTrack continueTrack );

private:
    // if continueTrack is not set extract all
    // if continueTrack is set - extract until reach it or closed, or border faced
    IsoLine extractOneLine_( const MeshEdgePoint& first, ContinueTrack continueTrack = {} );
    MeshEdgePoint toEdgePoint_( EdgeId e, float vo, float vd ) const;
    std::optional<MeshEdgePoint> findNextEdgePoint_( EdgeId e ) const;

private:
    
    const MeshTopology & topology_;
    const FaceBitSet * region_ = nullptr;
    ValueInVertex valueInVertex_;
    UndirectedEdgeBitSet seenEdges_;
};

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
        float vo = valueInVertex_( o );
        float vd = valueInVertex_( d );
        if ( vo < 0 && 0 <= vd )
            res.push_back( extractOneLine_( toEdgePoint_( e, vo, vd ) ) );
        else if ( vd < 0 && 0 <= vo )
            res.push_back( extractOneLine_( toEdgePoint_( e.sym(), vd, vo ) ) );
    }
    return res;
}

IsoLine Isoliner::track( const MeshTriPoint& start, ContinueTrack continueTrack )
{
    auto testEdge = [&] ( EdgeId e )->std::optional<MeshEdgePoint>
    {
        VertId o = topology_.org( e );
        VertId d = topology_.dest( e );
        float vo = valueInVertex_( o );
        float vd = valueInVertex_( d );
        if ( vd < 0 && 0 <= vo )
            return  toEdgePoint_( e.sym(), vd, vo );
        return {};
    };

    std::optional<MeshEdgePoint> startEdgePoint;
    if ( auto v = start.inVertex( topology_ ) )
    {
        for ( auto e : orgRing( topology_, v ) )
            if ( startEdgePoint = testEdge( e ) )
                break;
    }
    else if ( auto eOp = start.onEdge( topology_ ) )
    {
        if ( !( startEdgePoint = testEdge( eOp->e ) ) )
            startEdgePoint = eOp->sym();
        startEdgePoint = findNextEdgePoint_( startEdgePoint->e ); // `start` is first
    }
    else
    {
        for ( auto e : leftRing( topology_, start.e ) )
            if ( startEdgePoint = testEdge( e ) )
                break;
    }
    if ( !startEdgePoint )
        return {};
    return extractOneLine_( *startEdgePoint, continueTrack );
}

inline MeshEdgePoint Isoliner::toEdgePoint_( EdgeId e, float vo, float vd ) const
{
    assert( ( vo < 0 && 0 <= vd ) || ( vd < 0 && 0 <= vo ) );
    const float x = vo / ( vo - vd );
    return MeshEdgePoint( e, x );
}

std::optional<MeshEdgePoint> Isoliner::findNextEdgePoint_( EdgeId e ) const
{
    if ( !topology_.isLeftInRegion( e, region_ ) )
        return {};
    VertId o, d, x;
    topology_.getLeftTriVerts( e, o, d, x );
    const float vo = valueInVertex_( o );
    const float vd = valueInVertex_( d );
    const float vx = valueInVertex_( x );
    assert( ( vo < 0 && 0 <= vd ) || ( vd < 0 && 0 <= vo ) );
    if ( ( vo < 0 && vx < 0 ) || ( vd < 0 && vx >= 0 ) )
        return toEdgePoint_( topology_.prev( e.sym() ).sym(), vx, vd );
    else
        return toEdgePoint_( topology_.next( e ), vo, vx );
}

IsoLine Isoliner::extractOneLine_( const MeshEdgePoint & first, ContinueTrack continueTrack )
{
    std::vector<MeshEdgePoint> res;
    res.push_back( first );
    if ( continueTrack && !continueTrack( first ) )
        return res;
    seenEdges_.autoResizeSet( first.e.undirected() );

    bool closed = false;
    while ( auto next = findNextEdgePoint_( res.back().e ) )
    {
        if ( first.e == next->e )
        {
            res.push_back( first );
            if ( continueTrack )
                continueTrack( first );
            closed = true;
            break;
        }
        res.push_back( *next );
        if ( continueTrack && !continueTrack( *next ) )
            return res;
        seenEdges_.autoResizeSet( next->e.undirected() );
    }

    if ( continueTrack )
        return res;

    if ( !closed )
    {
        auto firstSym = first;
        firstSym = firstSym.sym(); // go backward
        std::vector<MeshEdgePoint> back;
        back.push_back( firstSym );
        while ( auto next = findNextEdgePoint_( back.back().e ) )
        {
            back.push_back( *next );
            seenEdges_.autoResizeSet( next->e.undirected() );
        }
        std::reverse( back.begin(), back.end() );
        back.pop_back(); // remove extra copy of firstSym
        for ( auto & i : back )
            i = i.sym(); // make consistent edge orientations of forward and backward passes
        res.insert( res.begin(), back.begin(), back.end() );
    }

    return res;
}

IsoLines extractIsolines( const MeshTopology & topology, 
    const Vector<float,VertId> & vertValues, float isoValue, const FaceBitSet * region )
{
    MR_TIMER;

    Isoliner s( topology, [&]( VertId v ) { return vertValues[v] - isoValue; }, region );
    return s.extract();
}

PlaneSections extractPlaneSections( const MeshPart & mp, const Plane3f & plane )
{
    MR_TIMER;

    Isoliner s( mp.mesh.topology, [&]( VertId v ) { return plane.distance( mp.mesh.points[v] ); }, mp.region );
    return s.extract();
}

PlaneSection trackSection( const MeshPart& mp, 
    const MeshTriPoint& start, MeshTriPoint& end, const Vector3f& direction, float distnace )
{
    MR_TIMER;
    if ( distnace == 0.0f )
    {
        end = start;
        return {};
    }
    const auto dir = distnace > 0.0f ? direction : -direction;
    distnace = std::abs( distnace );
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
        distnace -= dist;
        if ( distnace < 0.0f )
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
    if ( closed && distnace > 0.0f )
    {
        res.resize( int( res.size() ) - 1 );
        end = start;
        return res;
    }
    auto lastEdgePoint = res.back();
    auto lastPoint = mp.mesh.edgePoint( lastEdgePoint );
    res.resize( int( res.size() ) - 1 );
    auto lastSegmentLength = ( lastPoint - prevPoint ).length();
    float ratio = ( lastSegmentLength + distnace ) / lastSegmentLength;
    auto endPoint = ( 1.0f - ratio ) * prevPoint + ratio * lastPoint;
    end = mp.mesh.toTriPoint( mp.mesh.topology.right( lastEdgePoint.e ), endPoint );
    if ( closed &&
        dot( endPoint - prevPoint, lastPoint - prevPoint ) > dot( startPoint - prevPoint, lastPoint - prevPoint ) )
        end = start;
    return res;
}

Contour2f planeSectionToContour2f( const Mesh & mesh, const PlaneSection & section, const AffineXf3f & meshToPlane )
{
    MR_TIMER;
    Contour2f res;
    res.reserve( section.size() );
    for ( const auto & s : section )
    {
        auto p = meshToPlane( mesh.edgePoint( s ) );
        res.emplace_back( p.x, p.y );
    }

    return res;
}

Contours2f planeSectionsToContours2f( const Mesh & mesh, const PlaneSections & sections, const AffineXf3f & meshToPlane )
{
    MR_TIMER;
    Contours2f res;
    res.reserve( sections.size() );
    for ( const auto & s : sections )
        res.push_back( planeSectionToContour2f( mesh, s, meshToPlane ) );
    return res;
}

} //namespace MR
