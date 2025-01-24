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
#include "MRMeshTriPoint.h"
#include "MRFinally.h"
#include <atomic>

namespace MR
{

namespace
{

using ContinueTrack = std::function<bool( const MeshEdgePoint& )>;

template<typename V>
VertBitSet findNegativeVerts( const VertBitSet& vertRegion, V && valueInVertex )
{
    VertBitSet negativeVerts( vertRegion.size() );
    BitSetParallelFor( vertRegion, [&]( VertId v )
    {
        if ( valueInVertex( v ) < 0 )
            negativeVerts.set( v );
    } );
    return negativeVerts;
}

inline MeshEdgePoint toEdgePoint( EdgeId e, float vo, float vd )
{
    assert( ( vo < 0 && 0 <= vd ) || ( vd < 0 && 0 <= vo ) );
    const float x = vo / ( vo - vd );
    return MeshEdgePoint( e, x );
}

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
    IsoLines extract_();

private:
    const MeshTopology& topology_;
    const FaceBitSet* region_ = nullptr;
    VertMetric valueInVertex_;
    VertBitSet negativeVerts_;

    /// the edges crossed by the iso-line, but not yet extracted,
    /// filled in the beginning of extract() methods, and always null in track() method
    UndirectedEdgeBitSet activeEdges_;
};

void Isoliner::findNegativeVerts_()
{
    VertBitSet store;
    findNegativeVerts_( getIncidentVerts( topology_, region_, store ) );
}

void Isoliner::findNegativeVerts_( const VertBitSet& vertRegion )
{
    negativeVerts_ = findNegativeVerts( vertRegion, valueInVertex_ );
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

    return extract_();
}

IsoLines Isoliner::extract_()
{
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

    /// filter out the edges not crossed by the iso-line from activeEdges_
    BitSetParallelFor( activeEdges_, [&]( UndirectedEdgeId ue )
    {
        EdgeId e = ue;
        auto no = negativeVerts_.test( topology_.org( e ) );
        auto nd = negativeVerts_.test( topology_.dest( e ) );
        if ( no == nd )
            activeEdges_.reset( ue );
    } );

    return extract_();
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
    return MR::toEdgePoint( e, vo, vd );
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
    const bool activeEdgesEmpty = activeEdges_.empty();
    assert( activeEdgesEmpty || activeEdges_.test( first.undirected() ) );
    IsoLine res;
#ifndef NDEBUG
    MR_FINALLY{ assert( isConsistentlyOriented( topology_, res ) ); };
#endif // !NDEBUG
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
    
    assert( activeEdgesEmpty || activeEdges_.test( first.undirected() ) );
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
        if ( !activeEdgesEmpty && !activeEdges_.test( next.undirected() ) )
            return res; // the isoline left the region passed in extract( potentiallyCrossedEdges )
        if ( !addCrossedEdge( next ) )
            return res;
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
            assert( activeEdgesEmpty || activeEdges_.test( next.undirected() ) );
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

} //anonymous namespace

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

std::vector<TriangleSection> findTriangleSectionsByXYPlane( const MeshPart & mp, float zLevel )
{
    MR_TIMER
    auto levelInPoint = [&points = mp.mesh.points, zLevel] ( VertId v )
    {
        return points[v].z - zLevel;
    };

    VertBitSet store;
    const auto& regionVerts = getIncidentVerts( mp.mesh.topology, mp.region, store );
    const auto negativeVerts = findNegativeVerts( regionVerts, levelInPoint );

    FaceBitSet crossedFaces = mp.mesh.topology.getFaceIds( mp.region );
    BitSetParallelFor( crossedFaces, [&]( FaceId f )
    {
        auto vs = mp.mesh.topology.getTriVerts( f );
        int numNegative = negativeVerts.test( vs[0] ) + negativeVerts.test( vs[1] ) + negativeVerts.test( vs[2] );
        assert( numNegative >= 0 && numNegative <= 3 );
        if ( numNegative == 0 || numNegative == 3 )
            crossedFaces.reset( f );
    } );

    std::vector<TriangleSection> res;
    res.reserve( crossedFaces.count() );
    for ( auto f : crossedFaces )
        res.push_back( { .f = f } );
    ParallelFor( res, [&]( size_t i )
    {
        auto f = res[i].f;
        EdgeId e0, e1, e2;
        mp.mesh.topology.getTriEdges( f, e0, e1, e2 );
        const float z0 = levelInPoint( mp.mesh.topology.org( e0 ) );
        const float z1 = levelInPoint( mp.mesh.topology.org( e1 ) );
        const float z2 = levelInPoint( mp.mesh.topology.org( e2 ) );
        assert( z0 < 0 || z1 < 0 || z2 < 0 );
        assert( z0 >= 0 || z1 >= 0 || z2 >= 0 );
        LineSegm3f segm;
        int n = 0;
        auto checkEdge = [&]( EdgeId e, float vo, float vd )
        {
            if ( ( vo < 0 && 0 <= vd ) || ( vd < 0 && 0 <= vo ) )
            {
                auto p = mp.mesh.edgePoint( toEdgePoint( e, vo, vd ) );
                assert( n == 0 || n == 1 );
                if ( n == 0 )
                    segm.a = p;
                else
                    segm.b = p;
                ++n;
            }
        };
        assert( n == 2 );
        res[i].segm = segm;
    } );

    return res;
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

Expected<PlaneSection> trackSection( const MeshPart& mp, const MeshTriPoint& start, const MeshTriPoint& end, const Vector3f& planePoint, bool ccw )
{
    MR_TIMER;

    if ( fromSameTriangle( mp.mesh.topology, MeshTriPoint( start ), MeshTriPoint( end ) ) )
        return PlaneSection();

    auto startPoint = mp.mesh.triPoint( start );
    auto endPoint = mp.mesh.triPoint( end );
    auto crossDir = cross( startPoint - planePoint, endPoint - planePoint ).normalized();
    auto plane = Plane3f::fromDirAndPt( ccw ? crossDir : -crossDir, planePoint );
    VertMetric valueInVertex = [&] ( VertId v )
    {
        return plane.distance( mp.mesh.points[v] );
    };
    ContinueTrack continueTrack = [&] ( const MeshEdgePoint& next )->bool
    {
        return !fromSameTriangle( mp.mesh.topology, MeshTriPoint( next ), MeshTriPoint( end ) );
    };
    Isoliner s( mp.mesh.topology, valueInVertex, mp.region );
    auto res = s.track( start, continueTrack );
    if ( res.empty() )
    {
        assert( false );
        return unexpected( "Empty section" );
    }
    if ( res.size() > 1 && res.front() == res.back() )
        return unexpected( "Looped section" );
    if ( !fromSameTriangle( mp.mesh.topology, MeshTriPoint( res.back() ), MeshTriPoint( end ) ) )
        return unexpected( "Interrupted section" );
    return res;
}

bool isConsistentlyOriented( const MeshTopology & topology, const IsoLine & isoline )
{
    // works for both open and closed lines
    for ( int i = 0; i + 1 < isoline.size(); ++i )
    {
        auto l0 = topology.left( isoline[i].e );
        if ( !l0 )
            return false;
        auto r1 = topology.right( isoline[i+1].e );
        if ( l0 != r1 )
            return false;
    }
    return true;
}

FaceBitSet getCrossedFaces( const MeshTopology & topology, const IsoLine & isoline )
{
    assert( isConsistentlyOriented( topology, isoline ) );

    FaceBitSet res;
    for ( int i = 0; i + 1 < isoline.size(); ++i )
        res.autoResizeSet( topology.left( isoline[i].e ) );
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
