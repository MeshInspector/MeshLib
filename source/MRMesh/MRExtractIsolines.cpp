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
#include "MRMeshTriPoint.h"
#include "MRLineSegm3.h"
#include "MRTimer.h"
#include <atomic>

namespace MR
{

namespace
{

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

/// given linear function on edge by its values in two vertices with opposite signs,
/// finds point on edge, where the function is zero
inline MeshEdgePoint toEdgePoint( EdgeId e, float vo, float vd )
{
    assert( ( vo < 0 && 0 <= vd ) || ( vd < 0 && 0 <= vo ) );
    const float x = vo / ( vo - vd );
    return MeshEdgePoint( e, x );
}

/// given linear function on edge by value-getter for vertices (which must return values with opposite signs for edge's vertices),
/// finds point on edge, where the function is zero
template<class V>
inline MeshEdgePoint toEdgePoint( const MeshTopology & topology, V && v, EdgeId e )
{
    float vo = v( topology.org( e ) );
    float vd = v( topology.dest( e ) );
    return toEdgePoint( e, vo, vd );
}

/// Given boolean value isNegative defined in all vertices, and a start edge
/// with one negative and another not negative vertices, sequentially finds
/// other edges to the left of the current edge with the same property
template<typename N>
class Tracker
{
public:
    Tracker( const MeshTopology & topology, const N & isNegative, const FaceBitSet* region )
        : topology_( topology ), isNegative_( std::move( isNegative ) ), region_( region )
    {
    }

    void restart( EdgeId e )
    {
        assert( e );
        e_ = e;
        eOrgNeg_ = isNegative_( topology_.org( e_ ) );
        assert( eOrgNeg_ != isNegative_( topology_.dest( e_ ) ) );
    }

    EdgeId currEdge() const { return e_; }

    EdgeId findNextEdge()
    {
        if ( !topology_.isLeftInRegion( e_, region_ ) )
            return {};
        const VertId x = topology_.dest( topology_.next( e_ ) );
        const bool xNeg = isNegative_( x );

        if ( ( eOrgNeg_ && xNeg ) || ( !eOrgNeg_ && !xNeg ) )
        {
            eOrgNeg_ = xNeg;
            return e_ = topology_.prev( e_.sym() ).sym();
        }
        else
        {
            return e_ = topology_.next( e_ );
        }
    }

    template<class ContinueTrack>
    void track( const MeshTriPoint& start, const ContinueTrack& continueTrack );

private:
    const MeshTopology & topology_;
    N isNegative_;
    EdgeId e_;
    bool eOrgNeg_ = false;
    const FaceBitSet* region_ = nullptr;
};

auto isNegative( const VertBitSet & negativeVerts )
{
    return [&]( VertId v ) { return negativeVerts.test( v ); };
}

template<typename N>
template<class ContinueTrack>
void Tracker<N>::track( const MeshTriPoint& start, const ContinueTrack& continueTrack )
{
    auto testEdge = [&] ( EdgeId e ) -> EdgeId
    {
        VertId o = topology_.org( e );
        VertId d = topology_.dest( e );
        auto no = isNegative_( o );
        auto nd = isNegative_( d );
        return ( nd && !no ) ? e.sym() : EdgeId{};
    };

    EdgeId first;
    if ( auto v = start.inVertex( topology_ ) )
    {
        for ( auto e : orgRing( topology_, v ) )
        {
            if ( !topology_.isLeftInRegion( e, region_ ) )
                continue;
            auto te = topology_.prev( e.sym() ); // te has face with opposite (v) on left
            if ( auto se = testEdge( te ) )
            {
                first = se; // it has face with opposite (v) on right
                break;
            }
        }
    }
    else if ( auto eOp = start.onEdge( topology_ ) )
    {
        first = testEdge( eOp.e );
        if ( !first )
        {
            assert( testEdge( eOp.e.sym() ) );
            first = eOp.e;
        }
        restart( first );
        first = findNextEdge(); // first edge after (start)
    }
    else
    {
        for ( auto e : leftRing( topology_, start.e ) )
        {
            if ( auto se = testEdge( e ) )
            {
                first = se;
                break;
            }
        }
    }
    if ( !first )
        return;
    
    restart( first );
    do
    {
        if ( !continueTrack( e_ ) )
            break;
        if ( !findNextEdge() )
            break;
        assert( e_ );
    }
    while ( e_ != first );
}

class Isoliner
{
public:
    /// prepares to find iso-lines inside given region (or whole mesh if region==nullptr)
    Isoliner( const MeshTopology& topology, VertMetric valueInVertex, const FaceBitSet* region )
        : topology_( topology ), region_( region ), valueInVertex_( valueInVertex ), tracker_( topology_, isNegative( negativeVerts_ ), region_ )
        { findNegativeVerts_(); }

    /// prepares to find iso-lines crossing the edges in between given edges
    Isoliner( const MeshTopology& topology, VertMetric valueInVertex, const VertBitSet& vertRegion )
        : topology_( topology ), valueInVertex_( valueInVertex ), tracker_( topology_, isNegative( negativeVerts_ ), region_ )
        { findNegativeVerts_( vertRegion ); }

    /// if \param potentiallyCrossedEdges is given, then only these edges will be checked (otherwise all mesh edges)
    bool hasAnyLine( const UndirectedEdgeBitSet * potentiallyCrossedEdges = nullptr ) const;

    IsoLines extract();

    /// potentiallyCrossedEdges shall include all edges crossed by the iso-lines (some other edges there is permitted as well)
    IsoLines extract( UndirectedEdgeBitSet potentiallyCrossedEdges );

private:
    void findNegativeVerts_();
    void findNegativeVerts_( const VertBitSet& vertRegion );
    IsoLine extractOneLine_( EdgeId first );
    void computePointOnEachEdge_( IsoLine & line ) const;
    IsoLines extract_();

private:
    const MeshTopology& topology_;
    const FaceBitSet* region_ = nullptr;
    VertMetric valueInVertex_;
    VertBitSet negativeVerts_;
    Tracker<decltype( isNegative( negativeVerts_) )> tracker_;

    /// the edges crossed by the iso-line, but not yet extracted,
    /// filled in the beginning of extract() methods
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
    MR_TIMER
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

void Isoliner::computePointOnEachEdge_( IsoLine & line ) const
{
    ParallelFor( line, [&]( size_t i )
    {
        line[i] = toEdgePoint( topology_, valueInVertex_, line[i].e );
    } );
}

IsoLine Isoliner::extractOneLine_( EdgeId first )
{
    assert( activeEdges_.test( first.undirected() ) );
    IsoLine res;
    bool closed = false;

    tracker_.restart( first );
    EdgeId curr = first;
    for ( ;; )
    {
        if ( !activeEdges_.test_set( curr.undirected(), false ) )
            break; // the isoline left the region passed in extract( potentiallyCrossedEdges )
        res.push_back( MeshEdgePoint( curr, -1 ) );
        curr = tracker_.findNextEdge();
        if ( !curr )
            break;
        if ( first == curr )
        {
            res.push_back( MeshEdgePoint( first, -1 ) );
            closed = true;
            break;
        }
    }

    if ( !closed )
    {
        tracker_.restart( first.sym() ); // go backward
        IsoLine back;
        while ( auto next = tracker_.findNextEdge() )
        {
            if ( !activeEdges_.test( next.undirected() ) )
                break; // the isoline left the region passed in extract( potentiallyCrossedEdges )
            back.push_back( MeshEdgePoint( next, -1 ) );
            activeEdges_.reset( next.undirected() );
        }
        std::reverse( back.begin(), back.end() );
        for ( auto& i : back )
            i = i.sym(); // make consistent edge orientations of forward and backward passes
        res.insert( res.begin(), back.begin(), back.end() );
    }

    computePointOnEachEdge_( res );
    assert( isConsistentlyOriented( topology_, res ) );
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

std::vector<LineSegm3f> findTriangleSectionsByXYPlane( const MeshPart & mp, float zLevel,
    std::vector<FaceId> * faces, UseAABBTree u )
{
    MR_TIMER
    auto valueInPoint = [&points = mp.mesh.points, zLevel] ( VertId v )
    {
        return points[v].z - zLevel;
    };

    std::vector<FaceId> crossedFacesVec; // the faces crossing given zLevel

    if ( u == UseAABBTree::No || ( u == UseAABBTree::YesIfAlreadyConstructed && !mp.mesh.getAABBTreeNotCreate() ) )
    {
        // brute force checking all region triangles
        VertBitSet store;
        const auto& regionVerts = getIncidentVerts( mp.mesh.topology, mp.region, store );
        const auto negativeVerts = findNegativeVerts( regionVerts, valueInPoint );

        FaceBitSet crossedFaces = mp.mesh.topology.getFaceIds( mp.region );
        BitSetParallelFor( crossedFaces, [&]( FaceId f )
        {
            auto vs = mp.mesh.topology.getTriVerts( f );
            int numNegative = negativeVerts.test( vs[0] ) + negativeVerts.test( vs[1] ) + negativeVerts.test( vs[2] );
            assert( numNegative >= 0 && numNegative <= 3 );
            if ( numNegative == 0 || numNegative == 3 )
                crossedFaces.reset( f );
        } );

        crossedFacesVec.reserve( crossedFaces.count() );
        for ( auto f : crossedFaces )
            crossedFacesVec.push_back( f );
    }
    else
    {
        // optimized check using AABB tree
        xyPlaneMeshIntersect( mp, zLevel, nullptr, nullptr, nullptr, &crossedFacesVec );
    }

    std::vector<LineSegm3f> res( crossedFacesVec.size() );
    ParallelFor( res, [&]( size_t i )
    {
        auto vs = mp.mesh.topology.getTriVerts( crossedFacesVec[i] );
        float zs[3];
        int bs[3];
        for ( int j = 0; j < 3; ++j )
        {
            zs[j] = valueInPoint( vs[j] );
            bs[j] = int( zs[j] < 0 );
        }
        assert( bs[0] || bs[1] || bs[2] );
        assert( !bs[0] || !bs[1] || !bs[2] );

        // rotate vertices so that isoline intersects segments rvs[0]-rvs[1] and rvs[1]-rvs[2]
        VertId rvs[3];
        float rzs[3];
        if ( ( bs[2] ^ bs[0] ) == 0 )
        {
            rvs[0] = vs[0]; rvs[1] = vs[1]; rvs[2] = vs[2];
            rzs[0] = zs[0]; rzs[1] = zs[1]; rzs[2] = zs[2];
        }
        else if ( ( bs[0] ^ bs[1] ) == 0 )
        {
            rvs[0] = vs[1]; rvs[1] = vs[2]; rvs[2] = vs[0];
            rzs[0] = zs[1]; rzs[1] = zs[2]; rzs[2] = zs[0];
        }
        else if ( ( bs[1] ^ bs[2] ) == 0 )
        {
            rvs[0] = vs[2]; rvs[1] = vs[0]; rvs[2] = vs[1];
            rzs[0] = zs[2]; rzs[1] = zs[0]; rzs[2] = zs[1];
        }
        else
        {
            assert( false );
        }

        LineSegm3f segm;
        {
            assert ( ( rzs[0] < 0 && 0 <= rzs[1] ) || ( rzs[1] < 0 && 0 <= rzs[0] ) );
            const float x = rzs[0] / ( rzs[0] - rzs[1] );
            segm.a = x * mp.mesh.points[rvs[1]] + ( 1 - x ) * mp.mesh.points[rvs[0]];
        }
        {
            assert ( ( rzs[2] < 0 && 0 <= rzs[1] ) || ( rzs[1] < 0 && 0 <= rzs[2] ) );
            const float x = rzs[2] / ( rzs[2] - rzs[1] );
            segm.b = x * mp.mesh.points[rvs[1]] + ( 1 - x ) * mp.mesh.points[rvs[2]];
        }
        res[i] = segm;
    } );

    if ( faces )
        *faces = std::move( crossedFacesVec );

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
    auto valueInVertex = [&] ( VertId v )
    {
        return plane.distance( mp.mesh.points[v] );
    };

    PlaneSection res;
    auto continueTrack = [&] ( EdgeId e )
    {
        auto edgePoint = toEdgePoint( mp.mesh.topology, valueInVertex, e );
        res.push_back( edgePoint );
        auto point = mp.mesh.edgePoint( edgePoint );
        auto dist = ( point - prevPoint ).length();
        distance -= dist;
        if ( distance < 0.0f )
            return false;
        prevPoint = point;
        return true;
    };

    auto isNegative = [&] ( VertId v )
    {
        return valueInVertex( v ) < 0;
    };
    Tracker t( mp.mesh.topology, isNegative, mp.region );
    t.track( start, continueTrack );
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
    auto valueInVertex = [&] ( VertId v )
    {
        return plane.distance( mp.mesh.points[v] );
    };
    PlaneSection res;
    auto continueTrack = [&] ( EdgeId e )
    {
        auto next = toEdgePoint( mp.mesh.topology, valueInVertex, e );
        res.push_back( next );
        return !fromSameTriangle( mp.mesh.topology, MeshTriPoint( next ), MeshTriPoint( end ) );
    };
    auto isNegative = [&] ( VertId v )
    {
        return valueInVertex( v ) < 0;
    };
    Tracker t( mp.mesh.topology, isNegative, mp.region );
    t.track( start, continueTrack );
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
