#include "MR2DContoursTriangulation.h"
#include "MRMesh.h"
#include "MRVector.h"
#include "MRVector2.h"
#include "MRContour.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRConstants.h"
#include "MRMeshFixer.h"
#include "MREdgeIterator.h"
#include "MRMeshMetrics.h"
#include "MRMeshFillHole.h"
#include "MRMeshDelone.h"
#include <queue>
#include <algorithm>
#include <limits>

namespace MR
{

namespace PlanarTriangulation
{

class PlanarTriangulator
{
public:
    // constructor makes initial mesh which simply contain input contours as edges
    // (same vertices are merged and multiple edges are deleted)
    PlanarTriangulator( const Contours2d& contours, bool mergeClosePoints = true, bool abortWhenIntersect = false );
    // process line sweep queue and triangulate inside area of mesh (based on winding rule)
    std::optional<Mesh> run();
private:
    Mesh mesh_;
    bool mergeClosePoints_ = true;
    bool abortWhenIntersect_ = false;

    struct EdgeWindingInfo
    {
        int windingMod{ INT_MAX }; // modifier for merged edges (they can direct differently so we need to precalculate winding modifier)
        int winding{ INT_MAX };
        bool inside() const { return winding != 0 && winding != INT_MAX; } // may be other winding rule
    };
    Vector<EdgeWindingInfo, UndirectedEdgeId> windingInfo_;

    // struct to use easily compare mesh points by sweep line compare rule
    struct ComaparableVertId
    {
        ComaparableVertId( const Mesh* meshP, VertId v ) :mesh{ meshP }, id{ v }{}
        const Mesh* mesh;
        VertId id;
        bool operator<( const ComaparableVertId& other ) const;
        bool operator>( const ComaparableVertId& other ) const;
    };
    std::priority_queue<ComaparableVertId, std::vector<ComaparableVertId>, std::greater<ComaparableVertId>> queue_;

    // make base mesh only containing input contours as edge loops
    void initMeshByContours_( const Contours2d& contours );
    // merge same points on base mesh
    void mergeSamePoints_();
    void mergeSinglePare_( VertId unique, VertId same );

    // merging same vertices can make multiple edges, so clear it and update winding modifiers for merged edges
    void removeMultipleAfterMerge_();

    struct LoneRightmostLeft
    {
        EdgeId id;
        EdgeId upper;
        EdgeId lower;
    };
    // active edges - edges that currently intersect sweep line
    struct ActiveEdgeInfo
    {
        ActiveEdgeInfo( EdgeId e, float y ) :id{ e }, yPos{ y }{}
        EdgeId id;
        float yPos{ FLT_MAX };
        LoneRightmostLeft loneRightmostLeft;
    };
    LoneRightmostLeft lastLoneRightmostLeft_;
    std::vector<ActiveEdgeInfo> activeSweepEdges_;
    bool processOneVert_( VertId v );
    bool resolveIntersectios_();
};

bool PlanarTriangulator::ComaparableVertId::operator<( const ComaparableVertId& other ) const
{
    const auto& l = mesh->points[id];
    const auto& r = other.mesh->points[other.id];
    return l.x < r.x || ( l.x == r.x && l.y < r.y );
}

bool PlanarTriangulator::ComaparableVertId::operator>( const ComaparableVertId& other ) const
{
    const auto& l = mesh->points[id];
    const auto& r = other.mesh->points[other.id];
    return l.x > r.x || ( l.x == r.x && l.y > r.y );
}

PlanarTriangulator::PlanarTriangulator( const Contours2d& contours, bool mergeClosePoints /*= true*/, bool abortWhenIntersect /*= false*/ )
{
    abortWhenIntersect_ = abortWhenIntersect;
    mergeClosePoints_ = mergeClosePoints;
    initMeshByContours_( contours );
    mergeSamePoints_();
}

std::optional<Mesh> PlanarTriangulator::run()
{
    MR_TIMER;
    // process queue
    while ( !queue_.empty() )
    {
        auto active = queue_.top(); // cannot use std::move unfortunately since top() returns const reference
        queue_.pop();

        if ( !processOneVert_( active.id ) )
            return {};
    }
    // triangulate
    for ( auto e : undirectedEdges( mesh_.topology ) )
    {
        auto dirE = EdgeId( e << 1 );
        auto dotRes = dot( mesh_.edgeVector( dirE ), Vector3f::plusX() );
        if ( dotRes == 0.0f )
            continue;
        if ( dotRes < 0.0f )
            dirE = dirE.sym();
        if ( mesh_.topology.left( dirE ) )
            continue;
        if ( !windingInfo_[e].inside() )
            continue;
        FillHoleParams params;
        params.metric = getSimpleAreaMetric( mesh_, dirE );
        fillHole( mesh_, dirE, params );
    }
    makeDeloneEdgeFlips( mesh_, { .maxDeviationAfterFlip = std::numeric_limits<float>::epsilon() }, 100 );

    return std::move( mesh_ ); // move here to avoid copy of class member
}

void PlanarTriangulator::initMeshByContours_( const Contours2d& contours )
{
    MR_TIMER;
    int pointsSize = 0;
    for ( const auto& c : contours )
    {
        if ( c.size() > 3 )
        {
            assert( c.front() == c.back() );
            pointsSize += ( int( c.size() ) - 1 );
        }
    }
    mesh_.points.reserve( pointsSize );
    for ( const auto& c : contours )
    {
        if ( c.size() > 3 )
        {
            for ( int i = 0; i + 1 < c.size(); ++i )
                mesh_.addPoint( Vector3f{ float( c[i].x ),float( c[i].y ),0.0f } );
        }
    }
    int firstVert = 0;
    for ( const auto& c : contours )
    {
        if ( c.size() <= 3 )
            continue;

        int size = int( c.size() ) - 1;

        for ( int i = 0; i < size; ++i )
            mesh_.topology.setOrg( mesh_.topology.makeEdge(), VertId( firstVert + i ) );
        const auto& edgePerVert = mesh_.topology.edgePerVertex();
        for ( int i = 0; i < size; ++i )
            mesh_.topology.splice( edgePerVert[VertId( firstVert + i )], edgePerVert[VertId( firstVert + ( ( i + int( size ) - 1 ) % size ) )].sym() );
        firstVert += size;
    }
}

void PlanarTriangulator::mergeSamePoints_()
{
    MR_TIMER;
    std::vector<ComaparableVertId> sortedPoints;
    sortedPoints.reserve( mesh_.points.size() );
    for ( int i = 0; i < mesh_.points.size(); ++i )
        sortedPoints.emplace_back( &mesh_, VertId( i ) );
    std::sort( sortedPoints.begin(), sortedPoints.end() );

    int prevUnique = 0;
    for ( int i = 1; i < sortedPoints.size(); ++i )
    {
        if ( mesh_.points[sortedPoints[i].id] != mesh_.points[sortedPoints[prevUnique].id] )
        {
            prevUnique = i;
            continue;
        }
        // if same coords
        if ( mergeClosePoints_ )
            mergeSinglePare_( sortedPoints[prevUnique].id, sortedPoints[i].id );
    }

    removeMultipleAfterMerge_();

    for ( const auto& p : sortedPoints )
        if ( mesh_.topology.hasVert( p.id ) )
            queue_.push( p );
}

void PlanarTriangulator::mergeSinglePare_( VertId unique, VertId same )
{
    std::vector<EdgeId> sameEdges;
    bool sameToUniqueEdgeExists{ false };
    for ( auto eSame : orgRing( mesh_.topology, same ) )
    {
        sameEdges.push_back( eSame );
        if ( mesh_.topology.dest( eSame ) == unique )
            sameToUniqueEdgeExists = true;
    }

    if ( sameToUniqueEdgeExists )
    {
        // if part of same contour - no need to merge
        assert( sameEdges.size() == 2 );
        return;
    }

    for ( auto eSame : sameEdges )
    {
        auto sVec = mesh_.edgeVector( eSame ).normalized();
        float minAngle = std::numeric_limits<float>::max();
        EdgeId minEUnique;
        for ( auto eUnique : orgRing( mesh_.topology, unique ) )
        {
            auto uVec = mesh_.edgeVector( eUnique ).normalized();
            auto crossRes = cross( uVec, sVec );
            float angle = std::atan2( std::copysign( crossRes.length(), crossRes.z ), dot( uVec, sVec ) );
            if ( angle < 0.0f )
                angle += 2.0f * PI_F;
            if ( angle < minAngle )
            {
                minAngle = angle;
                minEUnique = eUnique;
            }
        }
        auto prev = mesh_.topology.prev( eSame );
        if ( prev != eSame )
            mesh_.topology.splice( prev, eSame );
        else
            mesh_.topology.setOrg( eSame, VertId{} );
        mesh_.topology.splice( minEUnique, eSame );
    }

}

void PlanarTriangulator::removeMultipleAfterMerge_()
{
    MR_TIMER;
    windingInfo_.resize( mesh_.topology.undirectedEdgeSize() );
    auto multiples = findMultipleEdges( mesh_.topology );
    for ( const auto& multiple : multiples )
    {
        std::vector<EdgeId> multiplesFromThis;
        for ( auto e : orgRing( mesh_.topology, multiple.first ) )
        {
            if ( mesh_.topology.dest( e ) == multiple.second )
                multiplesFromThis.push_back( e );
        }
        assert( multiplesFromThis.size() > 1 );

        auto& edgeInfo = windingInfo_[multiplesFromThis.front().undirected()];
        edgeInfo.windingMod = 1;
        bool uniqueIsOdd = int( multiplesFromThis.front() ) & 1;
        for ( int i = 1; i < multiplesFromThis.size(); ++i )
        {
            auto e = multiplesFromThis[i];
            bool isMEOdd = int( e ) & 1;
            edgeInfo.windingMod += ( ( uniqueIsOdd == isMEOdd ) ? 1 : -1 );
            mesh_.topology.splice( mesh_.topology.prev( e ), e );
            mesh_.topology.splice( mesh_.topology.prev( e.sym() ), e.sym() );
            assert( mesh_.topology.isLoneEdge( e ) );
        }
    }
}

bool PlanarTriangulator::processOneVert_( VertId v )
{
    // remove left, find right
    bool hasLeft = false;
    std::vector<ActiveEdgeInfo> rightGoingEdges;
    const auto& activePoint = mesh_.points[v];
    std::vector<int> indicesToRemoveFromActive;
    for ( auto e : orgRing( mesh_.topology, v ) )
    {
        auto lIt = std::find_if( activeSweepEdges_.begin(), activeSweepEdges_.end(), [e] ( const auto& a ) { return a.id == e.sym(); } );
        if ( lIt == activeSweepEdges_.end() )
            rightGoingEdges.emplace_back( e, activePoint.y );
        else
        {
            indicesToRemoveFromActive.push_back( int( std::distance( activeSweepEdges_.begin(), lIt ) ) );
            hasLeft = true;
        }
    }
    EdgeId lowestLeftEdge;
    if ( hasLeft )
    {
        // find lowest left for helper
        std::sort( indicesToRemoveFromActive.begin(), indicesToRemoveFromActive.end() );
        lowestLeftEdge = activeSweepEdges_[indicesToRemoveFromActive[0]].id.sym();
        // remove left
        for ( int i = int( indicesToRemoveFromActive.size() ) - 1; i >= 0; --i )
            activeSweepEdges_.erase( activeSweepEdges_.begin() + indicesToRemoveFromActive[i] );
    }

    // find correct place of right edges in active sweep edges
    int activeVPosition{ INT_MAX };// index of first edge, under activeV (INT_MAX - all edges are lower, -1 - all edges are upper)
    for ( int i = 0; i < activeSweepEdges_.size(); ++i )
    {
        auto org = mesh_.orgPnt( activeSweepEdges_[i].id );
        auto dest = mesh_.destPnt( activeSweepEdges_[i].id );
        auto xLength = dest.x - org.x;
        assert( xLength >= 0.0f );
        auto aXLength = activePoint.x - org.x;
        if ( xLength == 0.0f )
            activeSweepEdges_[i].yPos = ( org.y + dest.y ) * 0.5f; // vertical edge (as far as edges cannot intersect on the sweep line, we can safely use middle point as yPos )
        else
        {
            auto ratio = aXLength / xLength;
            activeSweepEdges_[i].yPos = ( 1 - ratio ) * org.y + ratio * dest.y;
        }
        if ( activePoint.y < activeSweepEdges_[i].yPos && activeVPosition == INT_MAX )
            activeVPosition = i - 1;
    }

    // find lowest rightGoingEdge (for correct insertion right edges into active sweep edges)
    auto findAngle = [&] ( const Vector3f& target, const Vector3f& baseVec )->float
    {
        auto edgeVec = ( target - mesh_.points[v] ).normalized();
        auto crossRes = cross( baseVec, edgeVec );
        float ang = std::atan2( std::copysign( crossRes.length(), crossRes.z ), dot( baseVec, edgeVec ) );
        if ( ang < 0.0f )
            ang += 2.0f * PI_F;
        return ang;
    };
    assert( hasLeft || !rightGoingEdges.empty() );
    int lowestRight = INT_MAX;
    float minAng = FLT_MAX;
    for ( int i = 0; i < rightGoingEdges.size(); ++i )
    {
        float ang = findAngle( mesh_.destPnt( rightGoingEdges[i].id ), Vector3f::minusY() );
        if ( ang < minAng )
        {
            minAng = ang;
            lowestRight = i;
        }
    }
    assert( rightGoingEdges.empty() || lowestRight != INT_MAX );

    bool hasOuter = activeVPosition != INT_MAX && activeVPosition != -1;
    // connect with outer contour if it has no left and inside (containing region should be internal)
    if ( !hasLeft && hasOuter && windingInfo_[activeSweepEdges_[activeVPosition].id.undirected()].inside() )
    {
        assert( lowestRight != INT_MAX );
        // find helper:
        // id of rightmost left vertex (it's lower edge) closest to active vertex
        // close to `helper` described here : https://www.cs.umd.edu/class/spring2020/cmsc754/Lects/lect05-triangulate.pdf
        EdgeId helperId;
        auto& upper = activeSweepEdges_[activeVPosition + 1];
        auto& lower = activeSweepEdges_[activeVPosition];
        if ( lastLoneRightmostLeft_.id && lastLoneRightmostLeft_.upper == upper.id && lastLoneRightmostLeft_.lower == lower.id )
        {
            helperId = lastLoneRightmostLeft_.id;
        }
        else
        {
            auto compUpper = ComaparableVertId( &mesh_, mesh_.topology.org( upper.id ) );
            auto compLower = ComaparableVertId( &mesh_, mesh_.topology.org( lower.id ) );
            if ( compUpper > compLower )
                helperId = mesh_.topology.prev( upper.id );
            else
                helperId = lower.id;
        }
        assert( helperId );
        if ( helperId == lastLoneRightmostLeft_.id )
            lastLoneRightmostLeft_.id = upper.loneRightmostLeft.id = lower.loneRightmostLeft.id = EdgeId{};
        auto newE = mesh_.topology.makeEdge();
        mesh_.topology.splice( helperId, newE );
        mesh_.topology.splice( mesh_.topology.prev( rightGoingEdges[lowestRight].id ), newE.sym() );
        windingInfo_.resize( newE.undirected() + 1 );
        windingInfo_[newE.undirected()].winding = 1; // mark inside
    }

    // connect rightmost left with no right edges to this edge, if needed
    {
        auto connect = [&] ( const LoneRightmostLeft& loneInfo ) mutable
        {
            const auto& orgPt = mesh_.orgPnt( loneInfo.id );

            Vector3f baseVec = ( orgPt - mesh_.points[v] ).normalized();
            float maxDiffAng = -FLT_MAX;
            EdgeId maxDiffE;
            for ( auto e : orgRing( mesh_.topology, v ) )
            {
                auto diffAng = findAngle( mesh_.destPnt( e ), baseVec );

                if ( diffAng > maxDiffAng )
                {
                    maxDiffAng = diffAng;
                    maxDiffE = e;
                }
            }
            auto newE = mesh_.topology.makeEdge();
            mesh_.topology.splice( loneInfo.id, newE );
            mesh_.topology.splice( maxDiffE, newE.sym() );

            windingInfo_.resize( newE.undirected() + 1 );
            windingInfo_[newE.undirected()].winding = 1; // mark inside
            if ( maxDiffE == lowestLeftEdge && activePoint.y > orgPt.y )
                lowestLeftEdge = newE.sym();
        };
        if ( activeVPosition != -1 && !activeSweepEdges_.empty() )
        {
            auto& lowerEdgeInfo = activeSweepEdges_[activeVPosition == INT_MAX ? int( activeSweepEdges_.size() - 1 ) : activeVPosition];
            if ( lowerEdgeInfo.loneRightmostLeft.id && lowerEdgeInfo.id == lowerEdgeInfo.loneRightmostLeft.lower )
            {
                connect( lowerEdgeInfo.loneRightmostLeft );
                lowerEdgeInfo.loneRightmostLeft.id = EdgeId{};
            }
        }
        if ( activeVPosition != INT_MAX && !activeSweepEdges_.empty() )
        {
            auto& upperEdgeInfo = activeSweepEdges_[activeVPosition + 1];
            if ( upperEdgeInfo.loneRightmostLeft.id && upperEdgeInfo.id == upperEdgeInfo.loneRightmostLeft.upper )
            {
                connect( upperEdgeInfo.loneRightmostLeft );
                upperEdgeInfo.loneRightmostLeft.id = EdgeId{};
            }
        }
        if ( lastLoneRightmostLeft_.id && 
            mesh_.topology.dest( lastLoneRightmostLeft_.upper ) == v &&
            mesh_.topology.dest( lastLoneRightmostLeft_.lower ) == v )
        {
            connect( lastLoneRightmostLeft_ );
            lastLoneRightmostLeft_.id = EdgeId{};
        }
    }

    // insert right going to active
    if ( !rightGoingEdges.empty() )
    {
        std::rotate( rightGoingEdges.begin(), rightGoingEdges.begin() + lowestRight, rightGoingEdges.end() );
        auto pos = activeVPosition == INT_MAX ? int( activeSweepEdges_.size() ) : activeVPosition + 1;
        activeSweepEdges_.insert( activeSweepEdges_.begin() + pos, rightGoingEdges.begin(), rightGoingEdges.end() );
    }
    else if ( hasOuter && windingInfo_[activeSweepEdges_[activeVPosition].id.undirected()].inside() )
    {
        assert( hasLeft );
        LoneRightmostLeft loneRightmostLeft;
        loneRightmostLeft.id = lowestLeftEdge;
        loneRightmostLeft.lower = activeSweepEdges_[activeVPosition].id;
        loneRightmostLeft.upper = activeSweepEdges_[activeVPosition + 1].id;

        activeSweepEdges_[activeVPosition].loneRightmostLeft = loneRightmostLeft;
        activeSweepEdges_[activeVPosition + 1].loneRightmostLeft = loneRightmostLeft;
        lastLoneRightmostLeft_ = std::move( loneRightmostLeft );
    }

    int windingLast = 0;
    // recalculate winding number for active edges
    for ( const auto& e : activeSweepEdges_ )
    {
        auto& info = windingInfo_[e.id.undirected()];
        if ( info.windingMod != INT_MAX )
            info.winding = windingLast + info.windingMod;
        else
            info.winding = windingLast + ( ( int( e.id ) & 1 ) ? -1 : 1 ); // even edges has same direction as original contour, but e.id always look to the right
        windingLast = info.winding;
    }

    // resolve intersections
    return resolveIntersectios_();
}

bool PlanarTriangulator::resolveIntersectios_()
{
    for ( int i = 0; i + 1 < activeSweepEdges_.size(); ++i )
    {
        auto org1 = mesh_.topology.org( activeSweepEdges_[i].id );
        auto dest1 = mesh_.topology.dest( activeSweepEdges_[i].id );
        auto org2 = mesh_.topology.org( activeSweepEdges_[i + 1].id );
        auto dest2 = mesh_.topology.dest( activeSweepEdges_[i + 1].id );
        bool canIntersect = org1 != org2 && dest1 != dest2;
        if ( !canIntersect )
            continue;

        auto p1 = mesh_.points[org1];
        auto p2 = mesh_.points[org2];
        auto d1 = mesh_.points[dest1] - p1;
        auto d2 = mesh_.points[dest2] - p2;

        auto n = cross( d1, d2 );
        if ( n == Vector3f() )
            continue; // parallel
        auto n1 = cross( d1, n );
        auto n2 = cross( d2, n );
        auto ratio1 = dot( p2 - p1, n2 ) / dot( d1, n2 );
        auto ratio2 = dot( p1 - p2, n1 ) / dot( d2, n1 );
        if ( ( ratio1 >= 1.0f || ratio1 <= 0.0f ) || ( ratio2 >= 1.0f || ratio2 <= 0.0f ) )
            continue; // no intersection

        if ( abortWhenIntersect_ )
            return false;

        auto intersection = p1 + ratio1 * d1;
        VertId vInter = mesh_.addPoint( intersection );
        // split 1
        auto pe1s = mesh_.topology.prev( activeSweepEdges_[i].id.sym() );
        mesh_.topology.splice( pe1s, activeSweepEdges_[i].id.sym() );
        auto e1n = mesh_.topology.makeEdge();
        if ( int( activeSweepEdges_[i].id ) & 1 )
            e1n = e1n.sym(); // new part direction should be same as old part's
        mesh_.topology.splice( pe1s, e1n.sym() );
        // split 2
        auto pe2s = mesh_.topology.prev( activeSweepEdges_[i + 1].id.sym() );
        mesh_.topology.splice( pe2s, activeSweepEdges_[i + 1].id.sym() );
        auto e2n = mesh_.topology.makeEdge();
        if ( int( activeSweepEdges_[i + 1].id ) & 1 )
            e2n = e2n.sym(); // new part direction should be same as old part's
        mesh_.topology.splice( pe2s, e2n.sym() );
        // connect all
        mesh_.topology.splice( activeSweepEdges_[i].id.sym(), e2n );
        mesh_.topology.splice( e2n, e1n );
        mesh_.topology.splice( e1n, activeSweepEdges_[i + 1].id.sym() );
        mesh_.topology.setOrg( e1n, vInter );

        // winding modifiers of new parts should be same as old parts'
        windingInfo_.resize( e2n.undirected() + 1 );
        windingInfo_[e1n.undirected()].windingMod = windingInfo_[activeSweepEdges_[i].id.undirected()].windingMod;
        windingInfo_[e2n.undirected()].windingMod = windingInfo_[activeSweepEdges_[i + 1].id.undirected()].windingMod;

        // update queue
        queue_.push( ComaparableVertId{ &mesh_,vInter } );
    }
    return true;
}

Mesh triangulateContours( const Contours2d& contours, bool mergeClosePoints /*= true*/ )
{
    PlanarTriangulator triangulator( contours, mergeClosePoints, false );
    auto res = triangulator.run();
    assert( res );
    if ( res )
        return std::move( *res );
    else
        return Mesh();
}

Mesh triangulateContours( const Contours2f& contours, bool mergeClosePoints /*= true*/ )
{
    const auto contsd = copyContours<Contours2d>( contours );
    PlanarTriangulator triangulator( contsd, mergeClosePoints, false );
    auto res = triangulator.run();
    assert( res );
    if ( res )
        return std::move( *res );
    else
        return Mesh();
}

std::optional<Mesh> triangulateDisjointContours( const Contours2d& contours, bool mergeClosePoints /*= true*/ )
{
    PlanarTriangulator triangulator( contours, mergeClosePoints, true );
    return triangulator.run();
}

std::optional<Mesh> triangulateDisjointContours( const Contours2f& contours, bool mergeClosePoints /*= true*/ )
{
    const auto contsd = copyContours<Contours2d>( contours );
    PlanarTriangulator triangulator( contsd, mergeClosePoints, true );
    return triangulator.run();
}

}

}