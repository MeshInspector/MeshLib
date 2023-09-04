#include "MR2DContoursTriangulation.h"
#include "MRMesh.h"
#include "MRVector.h"
#include "MRVector2.h"
#include "MRContour.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRConstants.h"
#include "MRRegionBoundary.h"
#include "MRMeshFixer.h"
#include "MREdgeIterator.h"
#include "MRMeshMetrics.h"
#include "MRMeshFillHole.h"
#include "MRMeshDelone.h"
#include "MRMeshCollidePrecise.h"
#include "MRBox.h"
#include "MR2to3.h"
#include <queue>
#include <algorithm>
#include <limits>
#include "MRPrecisePredicates2.h"

namespace MR
{

namespace PlanarTriangulation
{

class PlanarTriangulator
{
public:
    // constructor makes initial mesh which simply contain input contours as edges
    // if holesVertId is null - merge all vertices with same coordinates
    // otherwise only merge the ones with same initial vertId
    PlanarTriangulator( const Contours2d& contours, const HolesVertIds* holesVertId = nullptr, bool abortWhenIntersect = false );
    // process line sweep queue and triangulate inside area of mesh (based on winding rule)
    std::optional<Mesh> run();
private:
    Mesh mesh_;
    bool abortWhenIntersect_ = false;
    // this flag is set true if triangulation requires merging of two points that is forbidden
    bool incompleteMerge_ = false;

    struct EdgeWindingInfo
    {
        int windingMod{ INT_MAX }; // modifier for merged edges (they can direct differently so we need to precalculate winding modifier)
        int winding{ INT_MAX };
        bool inside() const { return winding != 0 && winding != INT_MAX; } // may be other winding rule
    };
    Vector<EdgeWindingInfo, UndirectedEdgeId> windingInfo_;

    CoordinateConverters2 converters_;
    // struct to use easily compare mesh points by sweep line compare rule
    struct ComaparableVertId
    {
        ComaparableVertId( const Mesh* meshP, VertId v, const CoordinateConverters2* conv ) :
            mesh{ meshP }, id{ v }, converters{ conv }{}
        const Mesh* mesh;
        VertId id;
        const CoordinateConverters2* converters;
        bool operator<( const ComaparableVertId& other ) const;
        bool operator>( const ComaparableVertId& other ) const;
    };
    std::priority_queue<ComaparableVertId, std::vector<ComaparableVertId>, std::greater<ComaparableVertId>> queue_;

    ComaparableVertId createCompVert_( VertId v );

    std::vector<EdgeId> findClosestCache_;
    int findClosestToFront_( std::vector<EdgeId>& edges, bool right );

    // make base mesh only containing input contours as edge loops
    void initMeshByContours_( const Contours2d& contours );
    // merge same points on base mesh
    void mergeSamePoints_( const HolesVertIds* holesVertId );
    void mergeSinglePare_( VertId unique, VertId same );

    void calculateWinding_();

    // merging same vertices can make multiple edges, so clear it and update winding modifiers for merged edges
    void removeMultipleAfterMerge_();

    void triangulateMonotoneBlock_( EdgeId holeEdgeId );
    std::vector<int> reflexChainCache_;

    struct LoneRightmostLeft
    {
        EdgeId id;
        EdgeId upper;
        EdgeId lower;
    };
    // active edges - edges that currently intersect sweep line
    struct ActiveEdgeInfo
    {
        ActiveEdgeInfo( EdgeId e ) :id{ e }{}
        EdgeId id;
        LoneRightmostLeft loneRightmostLeft;
    };
    std::vector<ActiveEdgeInfo> activeSweepEdges_;
    bool processOneVert_( VertId v );
    bool resolveIntersectios_();
};

bool PlanarTriangulator::ComaparableVertId::operator<( const ComaparableVertId& other ) const
{
    auto l = converters->toInt( to2dim( mesh->points[id] ) );
    auto r = converters->toInt( to2dim( other.mesh->points[other.id] ) );
    return l.x < r.x || ( l.x == r.x && l.y < r.y );
}

bool PlanarTriangulator::ComaparableVertId::operator>( const ComaparableVertId& other ) const
{
    auto l = converters->toInt( to2dim( mesh->points[id] ) );
    auto r = converters->toInt( to2dim( other.mesh->points[other.id] ) );
    return l.x > r.x || ( l.x == r.x && l.y > r.y );
}

PlanarTriangulator::PlanarTriangulator( const Contours2d& contours, const HolesVertIds* holesVertId /*= true*/, bool abortWhenIntersect /*= false*/ )
{
    abortWhenIntersect_ = abortWhenIntersect;
    initMeshByContours_( contours );

    auto box = Box3d( mesh_.computeBoundingBox() );
    converters_.toInt = [conv = getToIntConverter( box )]( const Vector2f& coord )
    {
        return to2dim( conv( to3dim( coord ) ) );
    };
    converters_.toFloat = [conv = getToFloatConverter( box )]( const Vector2i& coord )
    {
        return to2dim( conv( to3dim( coord ) ) );
    };

    mergeSamePoints_( holesVertId );
}

std::optional<Mesh> PlanarTriangulator::run()
{
    MR_TIMER;
    if ( incompleteMerge_ )
        return {};
    // process queue
    while ( !queue_.empty() )
    {
        auto active = queue_.top(); // cannot use std::move unfortunately since top() returns const reference
        queue_.pop();

        if ( !processOneVert_( active.id ) )
            return {};
    }
    // triangulate
    reflexChainCache_.reserve( 256 ); // reserve once to have less allocations later
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
    
        triangulateMonotoneBlock_( dirE );
    }
    makeDeloneEdgeFlips( mesh_, {}, 300 );

    return std::move( mesh_ ); // move here to avoid copy of class member
}

PlanarTriangulation::PlanarTriangulator::ComaparableVertId PlanarTriangulator::createCompVert_( VertId v )
{
    return ComaparableVertId( &mesh_, v, &converters_ );
}

int PlanarTriangulator::findClosestToFront_( std::vector<EdgeId>& edges, bool left )
{
    if ( edges.size() == 2 )
        return 1;
    std::array<PreciseVertCoords2, 3> pvc;
    auto org = mesh_.topology.org( edges[1] );
    pvc[2].id = org;
    pvc[2].pt = converters_.toInt( to2dim( mesh_.points[org] ) );
    PreciseVertCoords2 baseVertCoord;
    if ( edges[0] )
    {
        auto dest = mesh_.topology.dest( edges[0] );
        for ( int i = 1; i < edges.size(); ++i )
        {
            if ( dest == mesh_.topology.dest( edges[i] ) )
                return i;
        }
        baseVertCoord.id = dest;
        baseVertCoord.pt = converters_.toInt( to2dim( mesh_.points[dest] ) );
    }
    else
    {
        baseVertCoord.id = VertId{}; // -1
        baseVertCoord.pt = pvc[2].pt;
        baseVertCoord.pt.x -= 10000; // -X vec
    }
    auto getNextI = [&] ( int i, bool prev )
    {
        if ( prev )
        {
            if ( i == 1 )
                return int( edges.size() ) - 1;
            return i - 1;
        }
        else
        {
            if ( i == int( edges.size() ) - 1 )
                return 1;
            return i + 1;
        }
    };
    for ( int i = 1; ; )
    {
        pvc[0] = baseVertCoord;

        auto dest = mesh_.topology.dest( edges[i] );
        pvc[1].id = dest;
        pvc[1].pt = converters_.toInt( to2dim( mesh_.points[dest] ) );
        PreciseVertCoords2 coordI = pvc[1];

        bool ccwBI = ccw( pvc );
        int nextI = getNextI( i, ccwBI );

        dest = mesh_.topology.dest( edges[nextI] );
        pvc[1].id = dest;
        pvc[1].pt = converters_.toInt( to2dim( mesh_.points[dest] ) );

        bool ccwBIn = ccw( pvc );

        if ( ccwBI && !ccwBIn )
            return left ? i : nextI;
        if ( !ccwBI && ccwBIn )
            return left ? nextI : i;

        pvc[0] = coordI;
        bool ccwIIn = ccw( pvc );

        if ( ccwBI && ccwIIn )
            return left ? i : nextI;
        if ( !ccwBI && !ccwIIn )
            return left ? nextI : i;

        if ( nextI == 1 )
            break;
        i = nextI;
    }
    assert( false );
    return 0;
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

void PlanarTriangulator::mergeSamePoints_( const HolesVertIds* holesVertId )
{
    MR_TIMER;
    auto findRealVertId = [&] ( VertId patchId )
    {
        int holeId = 0;
        while ( patchId >= ( *holesVertId )[holeId].size() )
        {
            patchId -= int( ( *holesVertId )[holeId].size() );
            ++holeId;
        }
        return ( *holesVertId )[holeId][patchId];
    };
    std::vector<ComaparableVertId> sortedPoints;
    sortedPoints.reserve( mesh_.points.size() );
    for ( int i = 0; i < mesh_.points.size(); ++i )
        sortedPoints.emplace_back( createCompVert_( VertId( i ) ) );
    if ( !holesVertId )
        std::sort( sortedPoints.begin(), sortedPoints.end() );
    else
    {
        std::sort( sortedPoints.begin(), sortedPoints.end(), [&] ( const ComaparableVertId& l, const ComaparableVertId& r )
        {
            if ( l < r )
                return true;
            if ( l > r )
                return false;
            // find original vertId
            return findRealVertId( l.id ) < findRealVertId( r.id );
        } );
    }
    int prevUnique = 0;
    for ( int i = 1; i < sortedPoints.size(); ++i )
    {
        if ( mesh_.points[sortedPoints[i].id] != mesh_.points[sortedPoints[prevUnique].id] )
        {
            prevUnique = i;
            continue;
        }
        // if same coords
        if ( !holesVertId || findRealVertId( sortedPoints[prevUnique].id ) == findRealVertId( sortedPoints[i].id ) )
            mergeSinglePare_( sortedPoints[prevUnique].id, sortedPoints[i].id );
        else
            incompleteMerge_ = true;
    }

    removeMultipleAfterMerge_();

    for ( const auto& p : sortedPoints )
        if ( mesh_.topology.hasVert( p.id ) )
            queue_.push( p );
}

void PlanarTriangulator::mergeSinglePare_( VertId unique, VertId same )
{
    std::vector<EdgeId> sameEdges;
    int sameToUniqueEdgeIndex{ -1 };
    int i = 0;
    for ( auto eSame : orgRing( mesh_.topology, same ) )
    {
        sameEdges.push_back( eSame );
        if ( mesh_.topology.dest( eSame ) == unique )
        {
            assert( sameToUniqueEdgeIndex == -1 );
            sameToUniqueEdgeIndex = i;
        }
        ++i;
    }

    if ( sameToUniqueEdgeIndex != -1 )
    {
        // if part of same contour
        // disconnect before merge
        auto e = sameEdges[sameToUniqueEdgeIndex];
        mesh_.topology.splice( mesh_.topology.prev( e ), e );
        mesh_.topology.splice( mesh_.topology.prev( e.sym() ), e.sym() );
        sameEdges.erase( sameEdges.begin() + sameToUniqueEdgeIndex );
    }


    for ( auto eSame : sameEdges )
    {
        findClosestCache_.clear();
        findClosestCache_.push_back( eSame );
        for ( auto eUnique : orgRing( mesh_.topology, unique ) )
        {
            findClosestCache_.emplace_back( eUnique );
        }
        auto minEUnique = findClosestCache_[findClosestToFront_( findClosestCache_, false )];
        auto prev = mesh_.topology.prev( eSame );
        if ( prev != eSame )
            mesh_.topology.splice( prev, eSame );
        else
            mesh_.topology.setOrg( eSame, VertId{} );
        mesh_.topology.splice( minEUnique, eSame );
        if ( mesh_.topology.dest( minEUnique ) == mesh_.topology.dest( eSame ) )
        {
            auto& edgeInfo = windingInfo_.autoResizeAt( minEUnique.undirected() );
            if ( edgeInfo.windingMod == INT_MAX )
                edgeInfo.windingMod = 1;
            bool uniqueIsOdd = minEUnique.odd();
            bool sameIsOdd = eSame.odd();
            edgeInfo.windingMod += ( ( uniqueIsOdd == sameIsOdd ) ? 1 : -1 );
            mesh_.topology.splice( mesh_.topology.prev( eSame ), eSame );
            mesh_.topology.splice( mesh_.topology.prev( eSame.sym() ), eSame.sym() );
        }
    }

}

void PlanarTriangulator::calculateWinding_()
{
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
}

void PlanarTriangulator::removeMultipleAfterMerge_()
{
    MR_TIMER;
    windingInfo_.resize( mesh_.topology.undirectedEdgeSize() );
    auto multiples = findMultipleEdges( mesh_.topology ).value();
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

// find detailed explanation:
// https://www.cs.umd.edu/class/spring2020/cmsc754/Lects/lect05-triangulate.pdf
void PlanarTriangulator::triangulateMonotoneBlock_( EdgeId holeEdgeId )
{
    MR_TIMER;
    auto holeLoop = trackRightBoundaryLoop( mesh_.topology, holeEdgeId );
    auto lessPred = [&] ( EdgeId l, EdgeId r )
    {
        return createCompVert_( mesh_.topology.org( l ) ) < createCompVert_( mesh_.topology.org( r ) );
    };
    auto minMaxIt = std::minmax_element( holeLoop.begin(), holeLoop.end(), lessPred );

    int loopSize = int( holeLoop.size() );
    int minIndex = int( std::distance( holeLoop.begin(), minMaxIt.first ) );
    int maxIndex = int( std::distance( holeLoop.begin(), minMaxIt.second ) );
    auto nextLowerLoopInd = [&] ( int curIdx ) { return ( curIdx + 1 ) % loopSize; };
    auto nextUpperLoopInd = [&] ( int curIdx ) { return ( curIdx - 1 + loopSize ) % loopSize; };

    auto isReflex = [&] ( int prev, int cur, int next, bool lowerChain )
    {
        std::array<PreciseVertCoords2, 3> pvc;
        pvc[2].id = mesh_.topology.org( holeLoop[cur] );
        pvc[0].id = mesh_.topology.org( holeLoop[prev] );
        pvc[1].id = mesh_.topology.org( holeLoop[next] );
        for ( int i = 0; i < 3; ++i )
            pvc[i].pt = converters_.toInt( to2dim( mesh_.points[pvc[i].id] ) );
        return ccw( pvc ) == lowerChain;
    };

    auto addDiagonal = [&] ( int cur, int prev, bool lowerChain )->bool
    {
        auto& tp = mesh_.topology;
        if ( tp.prev( holeLoop[cur].sym() ) == holeLoop[prev] ||
            tp.next( holeLoop[cur] ).sym() == holeLoop[prev] )
        {
            tp.setLeft( holeLoop[cur], tp.addFaceId() );
            return true; // terminate
        }

        auto newE = tp.makeEdge();
        tp.splice( holeLoop[cur], newE );
        tp.splice( holeLoop[prev], newE.sym() );
        if ( lowerChain )
        {
            tp.setLeft( newE, tp.addFaceId() );
            holeLoop[prev] = newE.sym();
        }
        else
        {
            tp.setLeft( newE.sym(), tp.addFaceId() );
            holeLoop[cur] = newE;
        }
        return false;
    };

    int curIndex = minIndex;
    int curLower = minIndex;
    int curUpper = minIndex;

    auto& reflexChain = reflexChainCache_;
    reflexChain.resize( 0 );
    reflexChain.push_back( curIndex );
    bool reflexChainLower{ false };
    for ( ; ;)
    {
        assert( !reflexChain.empty() );
        // find current vertex on sweep line
        int nextLower = nextLowerLoopInd( curLower );
        int nextUpper = nextUpperLoopInd( curUpper );
        // assert that polygon is monotone
        assert( lessPred( holeLoop[curLower], holeLoop[nextLower] ) );
        assert( lessPred( holeLoop[curUpper], holeLoop[nextUpper] ) );
        bool currentOnLower = lessPred( holeLoop[nextLower], holeLoop[nextUpper] );
        if ( currentOnLower )
        {
            // shift by lower chain
            if ( curLower != maxIndex )
            {
                curIndex = nextLower;
                curLower = nextLower;
            }
        }
        else
        {
            // shift by upper chain
            if ( curUpper != maxIndex )
            {
                curIndex = nextUpper;
                curUpper = nextUpper;
            }
        }
        if ( curIndex == maxIndex )
        {
            currentOnLower = !reflexChainLower;
        }

        if ( reflexChain.size() == 1 ) // initial vertex
        {
            reflexChainLower = currentOnLower;
            reflexChain.push_back( curIndex );
            continue;
        }

        // process current vertex
        if ( currentOnLower == reflexChainLower ) // same chain Case 2
        {
            int prevChain = reflexChain[int( reflexChain.size() ) - 2];
            int curChain = reflexChain[int( reflexChain.size() ) - 1];
            while ( !isReflex( prevChain, curChain, curIndex, currentOnLower ) )
            {
                addDiagonal( curIndex, prevChain, currentOnLower );
                reflexChain.resize( int( reflexChain.size() ) - 1 );
                if ( reflexChain.size() < 2 )
                    break;
                prevChain = reflexChain[int( reflexChain.size() ) - 2];
                curChain = reflexChain[int( reflexChain.size() ) - 1];
            }
        }
        else // other chain Case 1
        {
            bool terminate = false;
            for ( int i = 1; i < reflexChain.size(); ++i )
            {
                assert( !terminate );
                terminate = addDiagonal( curIndex, reflexChain[i], currentOnLower );
            }
            if ( terminate )
                break;
            std::swap( reflexChain.front(), reflexChain.back() );
            reflexChain.resize( 1 );
            reflexChainLower = currentOnLower;
        }
        reflexChain.push_back( curIndex );
    }
}

bool PlanarTriangulator::processOneVert_( VertId v )
{
    // remove left, find right
    EdgeId anyLeftEdge;
    std::vector<ActiveEdgeInfo> rightGoingEdges;
    auto activePoint = converters_.toInt( to2dim( mesh_.points[v] ) );
    std::vector<int> indicesToRemoveFromActive;
    for ( auto e : orgRing( mesh_.topology, v ) )
    {
        auto lIt = std::find_if( activeSweepEdges_.begin(), activeSweepEdges_.end(), [e] ( const auto& a ) { return a.id == e.sym(); } );
        if ( lIt == activeSweepEdges_.end() )
            rightGoingEdges.emplace_back( e );
        else
        {
            indicesToRemoveFromActive.push_back( int( std::distance( activeSweepEdges_.begin(), lIt ) ) );
            anyLeftEdge = e;
        }
    }
    EdgeId lowestLeftEdge;
    if ( anyLeftEdge.valid() )
    {
        // also remove lone left after fixing duplicates
        for ( int i = 0; i < activeSweepEdges_.size(); ++i )
            if ( mesh_.topology.isLoneEdge( activeSweepEdges_[i].id ) )
                indicesToRemoveFromActive.push_back( i );
        // find lowest left for helper
        std::sort( indicesToRemoveFromActive.begin(), indicesToRemoveFromActive.end() );
        lowestLeftEdge = activeSweepEdges_[indicesToRemoveFromActive[0]].id.sym();
        // remove left
        for ( int i = int( indicesToRemoveFromActive.size() ) - 1; i >= 0; --i )
            activeSweepEdges_.erase( activeSweepEdges_.begin() + indicesToRemoveFromActive[i] );
    }

    // find correct place of right edges in active sweep edges
    int activeVPosition{ INT_MAX };// index of first edge, under activeV (INT_MAX - all edges are lower, -1 - all edges are upper)
    std::array<PreciseVertCoords2, 3> pvc;
    pvc[1].id = v;
    pvc[1].pt = converters_.toInt( to2dim( mesh_.points[pvc[1].id] ) );
    for ( int i = 0; i < activeSweepEdges_.size(); ++i )
    {
        pvc[0].id = mesh_.topology.org( activeSweepEdges_[i].id );
        pvc[2].id = mesh_.topology.dest( activeSweepEdges_[i].id );
        pvc[0].pt = converters_.toInt( to2dim( mesh_.points[pvc[0].id] ) );
        pvc[2].pt = converters_.toInt( to2dim( mesh_.points[pvc[2].id] ) );

        if ( activeVPosition == INT_MAX && ccw( pvc ) )
            activeVPosition = i - 1;
    }

    // find lowest rightGoingEdge (for correct insertion right edges into active sweep edges)
    assert( anyLeftEdge.valid() || !rightGoingEdges.empty() );

    int lowestRight = -1;
    if ( !rightGoingEdges.empty() )
    {
        findClosestCache_.clear();
        findClosestCache_.emplace_back( EdgeId{} ); // empty first min -x vec
        for ( int i = 0; i < rightGoingEdges.size(); ++i )
            findClosestCache_.push_back( rightGoingEdges[i].id );
        lowestRight = findClosestToFront_( findClosestCache_, true ) - 1;
    }
    assert( rightGoingEdges.empty() || lowestRight != -1 );

    bool hasOuter = activeVPosition != INT_MAX && activeVPosition != -1;
    // connect with outer contour if it has no left and inside (containing region should be internal)
    if ( !anyLeftEdge.valid() && hasOuter && windingInfo_[activeSweepEdges_[activeVPosition].id.undirected()].inside() )
    {
        assert( lowestRight != INT_MAX );
        // find helper:
        // id of rightmost left vertex (it's lower edge) closest to active vertex
        // close to `helper` described here : https://www.cs.umd.edu/class/spring2020/cmsc754/Lects/lect05-triangulate.pdf
        EdgeId helperId;
        auto& upper = activeSweepEdges_[activeVPosition + 1];
        auto& lower = activeSweepEdges_[activeVPosition];
        if ( upper.loneRightmostLeft.id && upper.loneRightmostLeft.id == lower.loneRightmostLeft.id &&
             upper.loneRightmostLeft.upper == upper.id && upper.loneRightmostLeft.lower == lower.id )
        {
            assert( lower.loneRightmostLeft.upper == upper.id && lower.loneRightmostLeft.lower == lower.id );
            helperId = upper.loneRightmostLeft.id;
        }
        else
        {
            auto compUpper = createCompVert_( mesh_.topology.org( upper.id ) );
            auto compLower = createCompVert_( mesh_.topology.org( lower.id ) );
            if ( compUpper > compLower )
                helperId = mesh_.topology.prev( upper.id );
            else
                helperId = lower.id;
        }
        assert( helperId );
        if ( helperId == upper.loneRightmostLeft.id )
            upper.loneRightmostLeft.id = lower.loneRightmostLeft.id = EdgeId{};
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
            findClosestCache_.clear();
            findClosestCache_.emplace_back( loneInfo.id.sym() );
            for ( auto e : orgRing( mesh_.topology, v ) )
                findClosestCache_.push_back( e );
            EdgeId maxDiffE = findClosestCache_[findClosestToFront_( findClosestCache_, false )];

            auto newE = mesh_.topology.makeEdge();
            mesh_.topology.splice( loneInfo.id, newE );
            mesh_.topology.splice( maxDiffE, newE.sym() );

            windingInfo_.resize( newE.undirected() + 1 );
            windingInfo_[newE.undirected()].winding = 1; // mark inside
            if ( maxDiffE == lowestLeftEdge )
            {
                std::array<PreciseVertCoords2, 3> pvc;
                pvc[2].id = v;
                pvc[2].pt = activePoint;
                pvc[0].id = mesh_.topology.dest( lowestLeftEdge );
                pvc[0].pt = converters_.toInt( to2dim( mesh_.points[pvc[0].id] ) );
                pvc[1].id = mesh_.topology.org( loneInfo.id );
                pvc[1].pt = converters_.toInt( to2dim( mesh_.points[pvc[1].id] ) );

                if ( ccw( pvc ) )
                    lowestLeftEdge = newE.sym();
            }
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
        assert( anyLeftEdge.valid() );
        LoneRightmostLeft loneRightmostLeft;
        loneRightmostLeft.id = lowestLeftEdge;
        loneRightmostLeft.lower = activeSweepEdges_[activeVPosition].id;
        loneRightmostLeft.upper = activeSweepEdges_[activeVPosition + 1].id;

        activeSweepEdges_[activeVPosition].loneRightmostLeft = loneRightmostLeft;
        activeSweepEdges_[activeVPosition + 1].loneRightmostLeft = loneRightmostLeft;
    }

    // resolve intersections
    if ( !resolveIntersectios_() )
        return false;

    calculateWinding_();
    return true;
}

bool PlanarTriangulator::resolveIntersectios_()
{
    std::array<PreciseVertCoords2, 4> pvc;
    for ( int i = 0; i + 1 < activeSweepEdges_.size(); ++i )
    {
        auto org1 = mesh_.topology.org( activeSweepEdges_[i].id );
        auto dest1 = mesh_.topology.dest( activeSweepEdges_[i].id );
        auto org2 = mesh_.topology.org( activeSweepEdges_[i + 1].id );
        auto dest2 = mesh_.topology.dest( activeSweepEdges_[i + 1].id );
        bool canIntersect = org1 != org2 && dest1 != dest2;
        if ( !canIntersect || !org1 || !org2 || !dest1 || !dest2 )
            continue;

        pvc[0].id = org1; pvc[1].id = dest1;
        pvc[2].id = org2; pvc[3].id = dest2;

        for ( int p = 0; p < 4; ++p )
            pvc[p].pt = converters_.toInt( to2dim( mesh_.points[pvc[p].id] ) );

        auto haveInter = doSegmentSegmentIntersect( pvc );
        if ( !haveInter.doIntersect )
            continue;

        if ( abortWhenIntersect_ )
            return false;

        auto intersection = findSegmentSegmentIntersectionPrecise(
            to2dim( mesh_.points[pvc[0].id] ), to2dim( mesh_.points[pvc[1].id] ),
            to2dim( mesh_.points[pvc[2].id] ), to2dim( mesh_.points[pvc[3].id] ),
            converters_ );
        
        VertId vInter = mesh_.addPoint( to3dim( intersection ) );
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
        PreciseVertCoords2 interPvc;
        interPvc.id = vInter;
        interPvc.pt = converters_.toInt( intersection );
        
        if (
            ( ccw( { pvc[0],interPvc,pvc[2] } ) == ccw( { interPvc,pvc[1],pvc[2] } ) ) &&
            ( ccw( { pvc[0],interPvc,pvc[3] } ) == ccw( { interPvc,pvc[1],pvc[3] } ) ) &&
            ( ccw( { pvc[2],interPvc,pvc[0] } ) == ccw( { interPvc,pvc[3],pvc[0] } ) ) &&
            ( ccw( { pvc[2],interPvc,pvc[1] } ) == ccw( { interPvc,pvc[3],pvc[1] } ) )
            )
        {
            mesh_.topology.splice( activeSweepEdges_[i].id.sym(), e2n );
            mesh_.topology.splice( e2n, e1n );
            mesh_.topology.splice( e1n, activeSweepEdges_[i + 1].id.sym() );
        }
        else
        {
            mesh_.topology.splice( activeSweepEdges_[i].id.sym(), e1n );
            mesh_.topology.splice( e1n, e2n );
            mesh_.topology.splice( e2n, activeSweepEdges_[i + 1].id.sym() );
        }
        mesh_.topology.setOrg( e1n, vInter );

        // winding modifiers of new parts should be same as old parts'
        windingInfo_.resize( e2n.undirected() + 1 );
        windingInfo_[e1n.undirected()].windingMod = windingInfo_[activeSweepEdges_[i].id.undirected()].windingMod;
        windingInfo_[e2n.undirected()].windingMod = windingInfo_[activeSweepEdges_[i + 1].id.undirected()].windingMod;
        
        if ( interPvc.pt == converters_.toInt( to2dim( mesh_.points[dest1] ) ) )
        {
            mergeSinglePare_( dest1, interPvc.id );
        }
        else if ( interPvc.pt == converters_.toInt( to2dim( mesh_.points[dest2] ) ) )
        {
            mergeSinglePare_( dest2, interPvc.id );
        }
        else
        {
            // update queue
            queue_.push( createCompVert_( vInter ) );
        }
    }
    return true;
}

HolesVertIds findHoleVertIdsByHoleEdges( const MeshTopology& tp, const std::vector<EdgePath>& holePaths )
{
    HolesVertIds res;
    res.reserve( holePaths.size() );
    for ( const auto& path : holePaths )
    {
        if ( path.size() < 3 )
            continue;
        res.emplace_back();
        auto& holeIds = res.back();
        holeIds.reserve( path.size() );
        for ( const auto& e : path )
            holeIds.emplace_back( tp.org( e ) );
    }
    return res;
}

Mesh triangulateContours( const Contours2d& contours, const HolesVertIds* holeVertsIds /*= nullptr*/ )
{
    PlanarTriangulator triangulator( contours, holeVertsIds, false );
    auto res = triangulator.run();
    assert( res );
    if ( res )
        return std::move( *res );
    else
        return Mesh();
}

Mesh triangulateContours( const Contours2f& contours, const HolesVertIds* holeVertsIds /*= nullptr*/ )
{
    const auto contsd = copyContours<Contours2d>( contours );
    PlanarTriangulator triangulator( contsd, holeVertsIds, false );
    auto res = triangulator.run();
    assert( res );
    if ( res )
        return std::move( *res );
    else
        return Mesh();
}

std::optional<Mesh> triangulateDisjointContours( const Contours2d& contours, const HolesVertIds* holeVertsIds /*= nullptr*/ )
{
    PlanarTriangulator triangulator( contours, holeVertsIds, true );
    return triangulator.run();
}

std::optional<Mesh> triangulateDisjointContours( const Contours2f& contours, const HolesVertIds* holeVertsIds /*= nullptr*/ )
{
    const auto contsd = copyContours<Contours2d>( contours );
    PlanarTriangulator triangulator( contsd, holeVertsIds, true );
    return triangulator.run();
}

}

}