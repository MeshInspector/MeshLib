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
#include "MRBitSetParallelFor.h"
#include "MRPrecisePredicates2.h"
#include "MRGTest.h"
#include <queue>
#include <algorithm>
#include <limits>

namespace MR
{

namespace PlanarTriangulation
{

int findClosestToFront( const MeshTopology& tp, const Vector<Vector3i, VertId>& pts,
    const std::vector<EdgeId>& edges, bool left )
{
    if ( edges.size() == 2 )
        return 1;
    std::array<PreciseVertCoords2, 3> pvc;
    auto org = tp.org( edges[1] );
    pvc[2].id = org;
    pvc[2].pt = to2dim( pts[org] );
    PreciseVertCoords2 baseVertCoord;
    if ( edges[0] )
    {
        auto dest = tp.dest( edges[0] );
        for ( int i = 1; i < edges.size(); ++i )
        {
            if ( dest == tp.dest( edges[i] ) )
                return i;
        }
        baseVertCoord.id = dest;
        baseVertCoord.pt = to2dim( pts[dest] );
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

        auto dest = tp.dest( edges[i] );
        pvc[1].id = dest;
        pvc[1].pt = to2dim( pts[dest] );
        PreciseVertCoords2 coordI = pvc[1];

        bool ccwBI = ccw( pvc );
        int nextI = getNextI( i, ccwBI );

        dest = tp.dest( edges[nextI] );
        pvc[1].id = dest;
        pvc[1].pt = to2dim( pts[dest] );

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

class SweepLineQueue
{
public:
    // constructor makes initial mesh which simply contain input contours as edges
    // if holesVertId is null - merge all vertices with same coordinates
    // otherwise only merge the ones with same initial vertId
    SweepLineQueue(
        const Contours2d& contours,
        const HolesVertIds* holesVertId = nullptr,
        bool abortWhenIntersect = false,
        WindingMode mode = WindingMode::NonZero,
        bool needOutline = false, // if set do not do real triangulation, just marks inside faces as present
        bool allowMerge = true, // one can disable merge for identical vertices, merge is useful on symbol contours
        std::vector<EdgePath>* outBoundaries = nullptr // optional out EdgePaths that corresponds to initial contours
        );

    size_t vertSize() const { return tp_.vertSize(); }
    std::optional<Mesh> run( IntersectionsMap* interMap = nullptr );

    bool findIntersections();
    void injectIntersections( IntersectionsMap* interMap );
    void makeMonotone();
    Mesh triangulate();
private:
    MeshTopology tp_;
    Vector<Vector3i, VertId> pts_;
    CoordinateConverters2 converters_;

    bool less_( VertId l, VertId r ) const
    {
        return std::tuple( pts_[l].x, pts_[l].y, l ) < std::tuple( pts_[r].x, pts_[r].y, r );
    }

// INITIALIZATION CLASS BLOCK
    // if set only marks inside faces as present (for further finding outline)
    bool needOutline_ = false;
    bool allowMerge_ = true;
    // if set fails on first found intersection
    bool abortWhenIntersect_ = false;
    // optional out EdgePaths that corresponds to initial contours
    std::vector<EdgePath>* outBoundaries_ = nullptr;
    // make base mesh only containing input contours as edge loops
    void initMeshByContours_( const Contours2d& contours );
    // merge same points on base mesh
    void mergeSamePoints_( const HolesVertIds* holesVertId );
    void mergeSinglePare_( VertId unique, VertId same );

    // merging same vertices can make multiple edges, so clear it and update winding modifiers for merged edges
    void removeMultipleAfterMerge_();

    enum Stage
    {
        Init, // start stage of class
        Intersections, // stage of finding and injecting intersections
        Monotonation, // stage of separating monotone polygons and filling winding numbers
        Triangulation // stage of triangulation of monotone blocks
    } stage_{ Init };

// MONOTONATION and TRIANGULATION CLASS BLOCK
    WindingMode windingMode_{ WindingMode::NonZero };

    struct EdgeWindingInfo
    {
        bool rightGoing{ false };
        int windingModifier{ INT_MAX }; // modifier for merged edges (they can direct differently so we need to precalculate winding modifier)
        int winding{ INT_MAX };
        bool inside( WindingMode mode ) const
        {
            if ( winding == INT_MAX )
                return false;
            if ( mode == WindingMode::NonZero )
                return winding != 0;
            else if ( mode == WindingMode::Positive )
                return winding > 0;
            else if ( mode == WindingMode::Negative )
                return winding < 0;
            return false;
        }

        EdgeWindingInfo() {} // Make `Vector` notice register the default constructor. :/
    };
    Vector<EdgeWindingInfo, UndirectedEdgeId> windingInfo_;

    void calculateWinding_();

    std::vector<int> reflexChainCache_;
    void triangulateMonotoneBlock_( EdgeId holeEdgeId );

// INTERSECTION CLASS BLOCK
    struct Intersection
    {
        EdgeId lower;
        EdgeId upper;
        VertId vId;
    };
    std::vector<Intersection> intersections_;

    void setupStartVertices_();
    // sorted vertices with no left-going edges
    std::vector<VertId> startVerts_;
    std::vector<EdgeId> startVertLowestRight_;
    // index of next `startVerts_`
    int startVertIndex_{ 0 };

    // sorted vertices
    std::vector<VertId> sortedVerts_;
    // index of next `startVerts_`
    int sortedVertIndex_{ 0 };

    struct SweepEdgeInfo
    {
        EdgeId edgeId;
        union Info
        {
            VertId interVertId{}; // without {} here, GCC produces wrong code: https://stackoverflow.com/q/68881411/7325599
            EdgeId loneEdgeId;
        };
        Info lowerInfo;
        Info upperInfo;
    };
    // edges that are intersected by sweep line ordered by position
    std::vector<SweepEdgeInfo> activeSweepEdges_;

    enum class EventType
    {
        Start, // item from `startVerts_`
        Destination, // one of the `activeSweepEdges_` destination vertices
        Intersection // intersection of two edges from `activeSweepEdges_`
    };
    struct Event
    {
        // type of event
        EventType type{ EventType::Start };
        // EventType::Start - position to inject start edges
        // EventType::Destination - id of lowest edge (with this destenation) in `activeSweepEdges_`
        // EventType::Intersection - id of lowest edge (with this intersection) in `activeSweepEdges_`
        int index{ -1 }; // -1 means that we finished queue
        // return true if event is valid
        operator bool() const { return index != -1; }
    };
    // ordered events after intersection stage
    std::vector<Event> events_;
    // get next queue element
    Event getNext_();

    void invalidateIntersection_( int indexLower );
    bool isIntersectionValid_( int indexLower );

    std::vector<SweepEdgeInfo> rightGoingCache_;
    std::vector<EdgeId> findClosestCache_;
    int findStartIndex_();
    void updateStartRightGoingCache_();
    void processStartEvent_( int index );
    void processDestenationEvent_( int index );
    void processIntersectionEvent_( int index );

    struct IntersectionInfo
    {
        VertId vId;
        bool processed{ false };
        operator bool() const { return vId.valid(); }
    };
    using IntersectionMap = HashMap<EdgePair, IntersectionInfo>;
    IntersectionMap intersectionsMap_; // needed to prevent recreation of same vertices multiple times
    void checkIntersection_( int index, bool lower );
    void checkIntersection_( int indexLower );
};

SweepLineQueue::SweepLineQueue(
    const Contours2d& contours,
    const HolesVertIds* holesVertId,
    bool abortWhenIntersect,
    WindingMode mode,
    bool needOutline,
    bool allowMerge,
    std::vector<EdgePath>* outBoundaries ) :
    needOutline_{ needOutline },
    allowMerge_{ allowMerge },
    abortWhenIntersect_{ abortWhenIntersect },
    outBoundaries_{ outBoundaries },
    windingMode_{ mode }
{
    Box3d box;
    for ( const auto& cont : contours )
        for ( const auto& p : cont )
            box.include( to3dim( p ) );

    converters_.toInt = [conv = getToIntConverter( box )] ( const Vector2f& coord )
    {
        return to2dim( conv( to3dim( coord ) ) );
    };
    converters_.toFloat = [conv = getToFloatConverter( box )] ( const Vector2i& coord )
    {
        return to2dim( conv( to3dim( coord ) ) );
    };

    initMeshByContours_( contours );
    mergeSamePoints_( holesVertId );
    setupStartVertices_();
}

std::optional<MR::Mesh> SweepLineQueue::run( IntersectionsMap* interMap )
{
    MR_TIMER;
    if ( !findIntersections() )
        return {};
    injectIntersections( interMap );
    makeMonotone();
    return triangulate();
}

bool SweepLineQueue::findIntersections()
{
    MR_TIMER;
    stage_ = Stage::Intersections;
    events_.clear();
    events_.reserve( tp_.numValidVerts() * 2 );
    while ( auto event = getNext_() )
    {
        if ( event.type == EventType::Start )
            processStartEvent_( event.index );
        else if ( event.type == EventType::Destination )
            processDestenationEvent_( event.index );
        else if ( event.type == EventType::Intersection )
        {
            if ( abortWhenIntersect_ )
                return false;
            processIntersectionEvent_( event.index );
        }
        events_.push_back( event );
    }
    return true;
}

void SweepLineQueue::injectIntersections( IntersectionsMap* interMap )
{
    MR_TIMER;

    if ( interMap )
        interMap->map.resize( intersections_.size() );

    windingInfo_.resize( windingInfo_.size() + intersections_.size() * 2 );
    Vector<EdgeId, UndirectedEdgeId> oldToFirstNewEdgeMap( tp_.undirectedEdgeSize() );

    if ( interMap )
    {
        // create mapping if needed
        for ( const auto& inter : intersections_ )
        {
            auto ind = size_t( inter.vId ) - interMap->shift;
            assert( ind < interMap->map.size() );
            auto& mapVal = interMap->map[ind];
            mapVal.lOrg = tp_.org( inter.lower );
            mapVal.lDest = tp_.dest( inter.lower );
            mapVal.uOrg = tp_.org( inter.upper );
            mapVal.uDest = tp_.dest( inter.upper );

            auto iP = converters_.toFloat( to2dim( pts_[inter.vId] ) );
            auto lO = converters_.toFloat( to2dim( pts_[mapVal.lOrg] ) );
            auto lD = converters_.toFloat( to2dim( pts_[mapVal.lDest] ) );
            auto uO = converters_.toFloat( to2dim( pts_[mapVal.uOrg] ) );
            auto uD = converters_.toFloat( to2dim( pts_[mapVal.uDest] ) );
            auto lVec = ( lD - lO );
            auto uVec = ( uD - uO );
            auto lVecLSq = lVec.lengthSq();
            auto uVecLSq = uVec.lengthSq();
            if ( lVecLSq == 0.0f )
                mapVal.lRatio = 0.0f;
            else
                mapVal.lRatio = std::clamp( dot( iP - lO, lD - lO ) / lVecLSq, 0.0f, 1.0f );

            if ( uVecLSq == 0.0f )
                mapVal.uRatio = 0.0f;
            else
                mapVal.uRatio = std::clamp( dot( iP - uO, uD - uO ) / uVecLSq, 0.0f, 1.0f );
        }
    }

    for ( const auto& inter : intersections_ )
    {
        // split edges
        // set new edge ids to the left and save old to the right
        // because of intersections order

        // prev lower
        auto pl = tp_.prev( inter.lower );
        // lower left
        auto ll = tp_.makeEdge();
        if ( inter.lower.odd() )
            ll = ll.sym(); // oddity should stay the same (for winding number)
        tp_.splice( pl, inter.lower );
        tp_.splice( pl, ll );
        tp_.splice( inter.lower, ll.sym() );

        // prev upper
        auto pu = tp_.prev( inter.upper );
        // upper left
        auto ul = tp_.makeEdge();
        if ( inter.upper.odd() )
            ul = ul.sym(); // oddity should stay the same (for winding number)

        tp_.splice( pu, inter.upper );
        tp_.splice( pu, ul );

        tp_.splice( inter.lower, ul.sym() );
        tp_.splice( ll.sym(), inter.upper );

        tp_.setOrg( inter.upper, inter.vId );

        // winding modifiers of new parts should be same as old parts
        windingInfo_[ll.undirected()].windingModifier = windingInfo_[inter.lower.undirected()].windingModifier;
        windingInfo_[ul.undirected()].windingModifier = windingInfo_[inter.upper.undirected()].windingModifier;

        auto& otfnL = oldToFirstNewEdgeMap[inter.lower.undirected()];
        if ( !otfnL )
            otfnL = ll;
        auto& otfnU = oldToFirstNewEdgeMap[inter.upper.undirected()];
        if ( !otfnU )
            otfnU = ul;
    }
    for ( auto& e : startVertLowestRight_ )
        if ( auto newE = oldToFirstNewEdgeMap[e.undirected()] )
            e = newE;
}

void SweepLineQueue::makeMonotone()
{
    MR_TIMER;
    stage_ = Stage::Monotonation;
    startVertIndex_ = 0;
    sortedVertIndex_ = 0;
    for ( auto event : events_ )
    {
        if ( event.type == EventType::Start )
            processStartEvent_( event.index );
        else
            processDestenationEvent_( event.index );
        calculateWinding_();
    }
}

Mesh SweepLineQueue::triangulate()
{
    MR_TIMER;
    stage_ = Stage::Triangulation;
    if ( !needOutline_ )
        reflexChainCache_.reserve( 256 ); // reserve once to have less allocations later
    for ( auto e : undirectedEdges( tp_ ) )
    {
        if ( e >= windingInfo_.size() )
            continue;
        const auto& windInfo = windingInfo_[e];
        if ( !windInfo.inside( windingMode_ ) )
            continue;
        auto dirE = EdgeId( e << 1 );
        if ( !windInfo.rightGoing )
            dirE = dirE.sym();
        if ( tp_.left( dirE ) )
            continue;

        if ( !needOutline_ )
            triangulateMonotoneBlock_( dirE ); // triangulate
        else
            tp_.setLeft( dirE, tp_.addFaceId() ); // mark present
    }
    Mesh mesh;
    mesh.topology = std::move( tp_ );
    mesh.points.resize( pts_.size() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
    {
        mesh.points[v] = to3dim(converters_.toFloat(to2dim(pts_[v])));
    } );
    if ( !needOutline_ )
    {
        makeDeloneEdgeFlips( mesh, {}, 300 );
    }
    return mesh;
}

void SweepLineQueue::setupStartVertices_()
{
    VertBitSet startVertices( tp_.vertSize() );
    BitSetParallelFor( tp_.getValidVerts(), [&] ( VertId v )
    {
        bool startVert = true;
        for ( auto e : orgRing( tp_, v ) )
        {
            if ( less_( tp_.dest( e ), v ) )
            {
                startVert = false;
                break;
            }
        }
        if ( startVert )
            startVertices.set( v );
    } );
    startVerts_.resize( startVertices.count() );
    startVertLowestRight_.resize( startVerts_.size() );
    int i = 0;
    for ( auto v : startVertices )
        startVerts_[i++] = v;

    std::sort( startVerts_.begin(), startVerts_.end(), [&] ( VertId l, VertId r )
    {
        return less_( l, r );
    } );
}

SweepLineQueue::Event SweepLineQueue::getNext_()
{
    Event outEvent;
    int minInterIndex = -1;

    VertId nextVertId;
    for ( ; sortedVertIndex_ < sortedVerts_.size();)
    {
        nextVertId = sortedVerts_[sortedVertIndex_];
        if ( tp_.hasVert( nextVertId ) )
            break;
        else
        {
            ++sortedVertIndex_;
            nextVertId = {};
        }
    }

    if ( !nextVertId )
        return outEvent;

    VertId minInter;
    VertId minDestId;
    for ( int i = 0; i < activeSweepEdges_.size(); ++i )
    {
        const auto& activeSweep = activeSweepEdges_[i];
        VertId destId = tp_.dest( activeSweep.edgeId );
        if ( !minDestId && destId == nextVertId )
        {
            minDestId = destId; // we need first
            outEvent.type = EventType::Destination;
            outEvent.index = i;
        }
        if ( stage_ != Stage::Intersections || !activeSweep.upperInfo.interVertId )
            continue;
        if ( !minInter || less_( activeSweep.upperInfo.interVertId, minInter ) )
        {
            minInter = activeSweep.upperInfo.interVertId;
            minInterIndex = i;
        }
    }

    if ( minInter )
    {
        if ( tp_.dest( activeSweepEdges_[minInterIndex].edgeId ) == nextVertId ||
            tp_.dest( activeSweepEdges_[minInterIndex + 1].edgeId ) == nextVertId ||
            less_( minInter, nextVertId ) )
        {
            outEvent.type = EventType::Intersection;
            outEvent.index = minInterIndex;
            nextVertId = {};
        }
    }

    if ( startVertIndex_ < startVerts_.size() )
    {
        if ( nextVertId == startVerts_[startVertIndex_] )
        {
            outEvent.type = EventType::Start;
            outEvent.index = findStartIndex_();
        }
    }

    return outEvent;
}

void SweepLineQueue::invalidateIntersection_( int indexLower )
{
    if ( indexLower >= 0 && indexLower < activeSweepEdges_.size() )
        activeSweepEdges_[indexLower].upperInfo.interVertId = {};
    if ( indexLower + 1 >= 0 && indexLower + 1 < activeSweepEdges_.size() )
        activeSweepEdges_[indexLower + 1].lowerInfo.interVertId = {};
}

bool SweepLineQueue::isIntersectionValid_( int indexLower )
{
    if ( indexLower < 0 || indexLower + 1 >= activeSweepEdges_.size() )
        return false;
    if ( !activeSweepEdges_[indexLower].upperInfo.interVertId )
        return false;
    return activeSweepEdges_[indexLower].upperInfo.interVertId == activeSweepEdges_[indexLower + 1].lowerInfo.interVertId;
}

int SweepLineQueue::findStartIndex_()
{
    int activeVPosition{ INT_MAX };// index of first edge, under activeV (INT_MAX - all edges are lower, -1 - all edges are upper)
    std::array<PreciseVertCoords2, 3> pvc;
    pvc[1].id = startVerts_[startVertIndex_];
    pvc[1].pt = to2dim( pts_[pvc[1].id] );
    for ( int i = 0; i < activeSweepEdges_.size(); ++i )
    {
        pvc[0].id = tp_.org( activeSweepEdges_[i].edgeId );
        pvc[2].id = tp_.dest( activeSweepEdges_[i].edgeId );
        pvc[0].pt = to2dim( pts_[pvc[0].id] );
        pvc[2].pt = to2dim( pts_[pvc[2].id] );

        if ( activeVPosition == INT_MAX && ccw( pvc ) )
            activeVPosition = i - 1;
    }

    return activeVPosition == INT_MAX ? int( activeSweepEdges_.size() ) : activeVPosition + 1;
}

void SweepLineQueue::updateStartRightGoingCache_()
{
    rightGoingCache_.clear();
    if ( stage_ == Stage::Intersections )
    {
        findClosestCache_.clear();
        findClosestCache_.emplace_back( EdgeId{} );
    }
    for ( auto e : orgRing( tp_, startVerts_[startVertIndex_] ) )
    {
        rightGoingCache_.emplace_back( SweepEdgeInfo{ .edgeId = e } );
        if ( stage_ == Stage::Intersections )
            findClosestCache_.push_back( e );
    }

    int pos = -1;
    if ( stage_ == Stage::Intersections )
    {
        pos = findClosestToFront( tp_, pts_, findClosestCache_, true ) - 1;
        assert( pos > -1 );
        startVertLowestRight_[startVertIndex_] = rightGoingCache_[pos].edgeId;
    }
    else
    {
        for ( int i = 0; i < rightGoingCache_.size(); ++i )
        {
            if ( rightGoingCache_[i].edgeId != startVertLowestRight_[startVertIndex_] )
                continue;
            pos = i;
            break;
        }
        assert( pos > -1 );
    }

    std::rotate( rightGoingCache_.begin(), rightGoingCache_.begin() + pos, rightGoingCache_.end() );
}

void SweepLineQueue::processStartEvent_( int index )
{
    updateStartRightGoingCache_();

    if ( stage_ == Stage::Intersections )
    {
        invalidateIntersection_( index - 1 );
    }

    if ( stage_ == Stage::Monotonation && index > 0 && index < activeSweepEdges_.size() &&
        windingInfo_[activeSweepEdges_[index - 1].edgeId.undirected()].inside( windingMode_ ) )
    {
        // find helper:
        // id of rightmost left vertex (it's lower edge) closest to active vertex
        // close to `helper` described here : https://www.cs.umd.edu/class/spring2020/cmsc754/Lects/lect05-triangulate.pdf
        EdgeId helperId;
        auto& lowerLone = activeSweepEdges_[index - 1].upperInfo.loneEdgeId;
        auto& upperLone = activeSweepEdges_[index].lowerInfo.loneEdgeId;
        assert( lowerLone == upperLone );
        if ( lowerLone )
        {
            helperId = lowerLone;
            lowerLone = upperLone = {};
        }
        else
        {
            auto lowerOrg = tp_.org( activeSweepEdges_[index - 1].edgeId );
            auto upperOrg = tp_.org( activeSweepEdges_[index].edgeId );
            if ( less_( lowerOrg, upperOrg ) )
                helperId = tp_.prev( activeSweepEdges_[index].edgeId );
            else
                helperId = activeSweepEdges_[index - 1].edgeId;
        }
        assert( helperId );

        auto newEdge = tp_.makeEdge();
        if ( activeSweepEdges_[index - 1].edgeId.odd() )
            newEdge = newEdge.sym();
        tp_.splice( helperId, newEdge );
        tp_.splice( rightGoingCache_.back().edgeId, newEdge.sym() );

        windingInfo_.autoResizeSet( newEdge.undirected(), windingInfo_[activeSweepEdges_[index - 1].edgeId.undirected()] );
    }

    activeSweepEdges_.insert( activeSweepEdges_.begin() + index, rightGoingCache_.begin(), rightGoingCache_.end() );

    if ( stage_ == Stage::Intersections )
    {
        checkIntersection_( index, true );
        checkIntersection_( index + 1, false );
    }

    ++startVertIndex_;
    ++sortedVertIndex_;
}

void SweepLineQueue::processDestenationEvent_( int index )
{
    int minIndex = index;
    int maxIndex = index;
    for ( int i = minIndex + 1; i < activeSweepEdges_.size(); ++i )
    {
        if ( tp_.dest( activeSweepEdges_[index].edgeId ) != tp_.dest( activeSweepEdges_[i].edgeId ) )
            break;
        maxIndex = i;
    }
    rightGoingCache_.clear();
    for ( auto e : orgRing0( tp_, activeSweepEdges_[minIndex].edgeId.sym() ) )
    {
        if ( e == activeSweepEdges_[maxIndex].edgeId.sym() )
            break;
        rightGoingCache_.emplace_back( SweepEdgeInfo{ .edgeId = e } );
    }
    int numLeft = maxIndex - minIndex + 1;
    int numRight = int( rightGoingCache_.size() );
    EdgeId lowestLeft = activeSweepEdges_[minIndex].edgeId;
    if ( stage_ == Stage::Monotonation )
    {
        // connect with prev lone if needed
        for ( int i = std::max( 0, minIndex - 1 ); i < std::min( maxIndex + 1, int( activeSweepEdges_.size() ) - 1 ); ++i )
        {
            auto& lowerLone = activeSweepEdges_[i].upperInfo.loneEdgeId;
            auto& upperLone = activeSweepEdges_[i + 1].lowerInfo.loneEdgeId;
            assert( lowerLone == upperLone );
            if ( !lowerLone )
                continue;

            EdgeId connectorEdgeId;
            if ( i < maxIndex )
                connectorEdgeId = activeSweepEdges_[i + 1].edgeId.sym();
            else
                connectorEdgeId = tp_.prev( activeSweepEdges_[i].edgeId.sym() );

            auto newEdge = tp_.makeEdge();
            if ( activeSweepEdges_[i].edgeId.odd() )
                newEdge = newEdge.sym();
            tp_.splice( lowerLone, newEdge );
            tp_.splice( connectorEdgeId, newEdge.sym() );

            lowerLone = upperLone = {};

            windingInfo_.autoResizeSet( newEdge.undirected(), windingInfo_[activeSweepEdges_[i].edgeId.undirected()] );
            if ( i == minIndex - 1 )
                lowestLeft = newEdge;
        }
    }
    if ( numRight == 0 )
    {
        if ( stage_ == Stage::Monotonation && minIndex > 0 && maxIndex + 1 < activeSweepEdges_.size() &&
            windingInfo_[activeSweepEdges_[minIndex - 1].edgeId.undirected()].inside( windingMode_ ) )
        {
            activeSweepEdges_[minIndex - 1].upperInfo.loneEdgeId = lowestLeft.sym();
            activeSweepEdges_[maxIndex + 1].lowerInfo.loneEdgeId = lowestLeft.sym();
        }
        activeSweepEdges_.erase( activeSweepEdges_.begin() + minIndex, activeSweepEdges_.begin() + maxIndex + 1 );
        if ( stage_ == Stage::Intersections )
        {
            checkIntersection_( minIndex - 1, false );
        }
    }
    else
    {
        for ( int i = minIndex; i < minIndex + std::min( numLeft, numRight ); ++i )
        {
            assert( i < activeSweepEdges_.size() );
            activeSweepEdges_[i] = rightGoingCache_[i - minIndex];
        }
        if ( numLeft > numRight )
            activeSweepEdges_.erase( activeSweepEdges_.begin() + minIndex + numRight, activeSweepEdges_.begin() + maxIndex + 1 );
        else if ( numLeft < numRight )
            activeSweepEdges_.insert( activeSweepEdges_.begin() + maxIndex + 1, rightGoingCache_.begin() + numLeft, rightGoingCache_.end() );

        if ( stage_ == Stage::Intersections )
        {
            checkIntersection_( minIndex + numRight, true );
            checkIntersection_( minIndex - 1, false );
        }
    }
    ++sortedVertIndex_;
}

void SweepLineQueue::processIntersectionEvent_( int index )
{
    bool isValid = isIntersectionValid_( index );
    if ( isValid )
    {
        intersections_.emplace_back( Intersection{
            .lower = activeSweepEdges_[index].edgeId,
            .upper = activeSweepEdges_[index + 1].edgeId } );
    }
    invalidateIntersection_( index );
    if ( !isValid )
        return;

    auto minEdgeId = std::min( activeSweepEdges_[index].edgeId, activeSweepEdges_[index + 1].edgeId );
    auto maxEdgeId = std::max( activeSweepEdges_[index].edgeId, activeSweepEdges_[index + 1].edgeId );

    auto& interInfo = intersectionsMap_.at( { minEdgeId,maxEdgeId } );
    assert( !interInfo.processed );
    interInfo.processed = true;
    intersections_.back().vId = interInfo.vId;

    invalidateIntersection_( index - 1 );
    invalidateIntersection_( index + 1 );

    std::swap( activeSweepEdges_[index], activeSweepEdges_[index + 1] );

    checkIntersection_( index, true );
    checkIntersection_( index + 1, false );
}

void SweepLineQueue::checkIntersection_( int index, bool lower )
{
    if ( index < 0 || index >= activeSweepEdges_.size() )
        return;
    if ( lower && index == 0 )
        return;
    if ( !lower && index + 1 == activeSweepEdges_.size() )
        return;
    if ( lower && index >= 1 )
        return checkIntersection_( index - 1 );
    if ( !lower && index + 1 < activeSweepEdges_.size() )
        return checkIntersection_( index );
}

void SweepLineQueue::checkIntersection_( int i )
{
    assert( i >= 0 && i + 1 < activeSweepEdges_.size() );

    // fill up
    std::array<PreciseVertCoords2, 4> pvc;
    auto org1 = tp_.org( activeSweepEdges_[i].edgeId );
    auto dest1 = tp_.dest( activeSweepEdges_[i].edgeId );
    auto org2 = tp_.org( activeSweepEdges_[i + 1].edgeId );
    auto dest2 = tp_.dest( activeSweepEdges_[i + 1].edgeId );
    bool canIntersect = org1 != org2 && dest1 != dest2;
    if ( !canIntersect || !org1 || !org2 || !dest1 || !dest2 )
        return;

    pvc[0].id = org1; pvc[1].id = dest1;
    pvc[2].id = org2; pvc[3].id = dest2;

    for ( int p = 0; p < 4; ++p )
        pvc[p].pt = to2dim( pts_[pvc[p].id] );

    auto haveInter = doSegmentSegmentIntersect( pvc );
    if ( !haveInter.doIntersect )
        return;

    auto minEdgeId = std::min( activeSweepEdges_[i].edgeId, activeSweepEdges_[i + 1].edgeId );
    auto maxEdgeId = std::max( activeSweepEdges_[i].edgeId, activeSweepEdges_[i + 1].edgeId );
    auto& interInfo = intersectionsMap_[{minEdgeId, maxEdgeId}];
    if ( !interInfo )
    {
        interInfo.vId = tp_.addVertId();
        pts_.autoResizeSet( interInfo.vId,
            to3dim( findSegmentSegmentIntersectionPrecise( pvc[0].pt, pvc[1].pt, pvc[2].pt, pvc[3].pt ) ) );
    }
    else if ( interInfo.processed )
        return;

    activeSweepEdges_[i].upperInfo.interVertId = interInfo.vId;
    activeSweepEdges_[i + 1].lowerInfo.interVertId = interInfo.vId;
}

void SweepLineQueue::initMeshByContours_( const Contours2d& contours )
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
    pts_.reserve( pointsSize );
    for ( const auto& c : contours )
    {
        if ( c.size() > 3 )
        {
            for ( int i = 0; i + 1 < c.size(); ++i )
            {
                VertId v = tp_.addVertId();
                pts_.autoResizeSet( v, to3dim( converters_.toInt( Vector2f( c[i] ) ) ) );
            }
        }
    }

    int boundId = -1;
    if ( outBoundaries_ )
        outBoundaries_->resize( contours.size() );

    int firstVert = 0;
    for ( const auto& c : contours )
    {
        ++boundId;
        if ( c.size() <= 3 )
            continue;

        int size = int( c.size() ) - 1;

        if ( outBoundaries_ )
            ( *outBoundaries_ )[boundId].resize( size );

        for ( int i = 0; i < size; ++i )
        {
            auto newEdgeId = tp_.makeEdge();
            tp_.setOrg( newEdgeId, VertId( firstVert + i ) );
            if ( outBoundaries_ )
                ( *outBoundaries_ )[boundId][i] = newEdgeId;
        }
        const auto& edgePerVert = tp_.edgePerVertex();
        for ( int i = 0; i < size; ++i )
            tp_.splice( edgePerVert[VertId( firstVert + i )], edgePerVert[VertId( firstVert + ( ( i + int( size ) - 1 ) % size ) )].sym() );
        firstVert += size;
    }
}

void SweepLineQueue::mergeSamePoints_( const HolesVertIds* holesVertId )
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
    sortedVerts_.reserve( pts_.size() );
    for ( int i = 0; i < pts_.size(); ++i )
        sortedVerts_.emplace_back( VertId( i ) );
    if ( !holesVertId )
        std::sort( sortedVerts_.begin(), sortedVerts_.end(), [&] ( VertId l, VertId r ) { return less_( l, r ); } );
    else
    {
        std::sort( sortedVerts_.begin(), sortedVerts_.end(), [&] ( VertId l, VertId r )
        {
            return std::tuple( pts_[l].x, pts_[l].y, findRealVertId( l ) ) < std::tuple( pts_[r].x, pts_[r].y, findRealVertId( r ) );
        } );
    }

    if ( !allowMerge_ )
    {
        windingInfo_.resize( tp_.undirectedEdgeSize() );
        return;
    }

    int prevUnique = 0;
    for ( int i = 1; i < sortedVerts_.size(); ++i )
    {
        bool sameIntCoord = pts_[sortedVerts_[i]] == pts_[sortedVerts_[prevUnique]];
        if ( !sameIntCoord )
        {
            prevUnique = i;
            continue;
        }
        // if same coords
        if ( !holesVertId || findRealVertId( sortedVerts_[prevUnique] ) == findRealVertId( sortedVerts_[i] ) )
            mergeSinglePare_( sortedVerts_[prevUnique], sortedVerts_[i] );
    }

    if ( holesVertId ) // sort with correct indices in case of other way sort before
        std::sort( sortedVerts_.begin(), sortedVerts_.end(), [&] ( VertId l, VertId r ) { return less_( l, r ); } );

    removeMultipleAfterMerge_();
}

void SweepLineQueue::mergeSinglePare_( VertId unique, VertId same )
{
    std::vector<EdgeId> sameEdges;
    int sameToUniqueEdgeIndex{ -1 };
    int i = 0;
    for ( auto eSame : orgRing( tp_, same ) )
    {
        sameEdges.push_back( eSame );
        if ( tp_.dest( eSame ) == unique )
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
        tp_.splice( tp_.prev( e ), e );
        tp_.splice( tp_.prev( e.sym() ), e.sym() );
        sameEdges.erase( sameEdges.begin() + sameToUniqueEdgeIndex );
    }

    for ( auto eSame : sameEdges )
    {
        findClosestCache_.clear();
        findClosestCache_.push_back( eSame );
        for ( auto eUnique : orgRing( tp_, unique ) )
        {
            findClosestCache_.emplace_back( eUnique );
        }
        auto minEUnique = findClosestCache_[findClosestToFront( tp_, pts_, findClosestCache_, false )];
        auto prev = tp_.prev( eSame );
        if ( prev != eSame )
            tp_.splice( prev, eSame );
        else
            tp_.setOrg( eSame, VertId{} );
        tp_.splice( minEUnique, eSame );
        if ( tp_.dest( minEUnique ) == tp_.dest( eSame ) )
        {
            auto& edgeInfo = windingInfo_.autoResizeAt( minEUnique.undirected() );
            if ( edgeInfo.windingModifier == INT_MAX )
                edgeInfo.windingModifier = 1;
            bool uniqueIsOdd = minEUnique.odd();
            bool sameIsOdd = eSame.odd();
            edgeInfo.windingModifier += ( ( uniqueIsOdd == sameIsOdd ) ? 1 : -1 );
            tp_.splice( tp_.prev( eSame ), eSame );
            tp_.splice( tp_.prev( eSame.sym() ), eSame.sym() );
        }
    }
}

void SweepLineQueue::removeMultipleAfterMerge_()
{
    MR_TIMER;
    windingInfo_.resize( tp_.undirectedEdgeSize() );
    auto multiples = findMultipleEdges( tp_ ).value();
    for ( const auto& multiple : multiples )
    {
        std::vector<EdgeId> multiplesFromThis;
        for ( auto e : orgRing( tp_, multiple.first ) )
        {
            if ( tp_.dest( e ) == multiple.second )
                multiplesFromThis.push_back( e );
        }
        assert( multiplesFromThis.size() > 1 );

        if ( outBoundaries_ )
        {
            auto& bounds = *outBoundaries_;
            auto getBoundId = [&bounds] ( EdgeId e )->std::pair<int, int>
            {
                int i0 = 0;
                auto i1 = int( e.undirected() );
                while ( i1 >= bounds[i0].size() )
                {
                    ++i0;
                    i1 -= int( bounds[i0].size() );
                }
                assert( e.undirected() == bounds[i0][i1].undirected() );
                return { i0,i1 };
            };
            auto [bf0, bf1] = getBoundId( multiplesFromThis.front() );
            auto bf = bounds[bf0][bf1];
            for ( int i = 1; i < multiplesFromThis.size(); ++i )
            {
                auto [bi0, bi1] = getBoundId( multiplesFromThis[i] );
                auto& bi = bounds[bi0][bi1];
                bi = multiplesFromThis[i] == bi ? bf : bf.sym();
            }
        }

        auto& edgeInfo = windingInfo_[multiplesFromThis.front().undirected()];
        edgeInfo.windingModifier = 1;
        bool uniqueIsOdd = int( multiplesFromThis.front() ) & 1;
        for ( int i = 1; i < multiplesFromThis.size(); ++i )
        {
            auto e = multiplesFromThis[i];
            bool isMEOdd = int( e ) & 1;
            edgeInfo.windingModifier += ( ( uniqueIsOdd == isMEOdd ) ? 1 : -1 );
            tp_.splice( tp_.prev( e ), e );
            tp_.splice( tp_.prev( e.sym() ), e.sym() );
            assert( tp_.isLoneEdge( e ) );
        }
    }
}

void SweepLineQueue::calculateWinding_()
{
    int windingLast = 0;
    // recalculate winding number for active edges
    for ( const auto& e : activeSweepEdges_ )
    {
        auto& info = windingInfo_[e.edgeId.undirected()];
        info.rightGoing = e.edgeId.even();
        if ( info.windingModifier != INT_MAX )
            info.winding = windingLast + info.windingModifier;
        else
            info.winding = windingLast + ( e.edgeId.odd() ? -1 : 1 ); // even edges has same direction as original contour, but e.id always look to the right
        windingLast = info.winding;
    }
}

// find detailed explanation:
// https://www.cs.umd.edu/class/spring2020/cmsc754/Lects/lect05-triangulate.pdf
void SweepLineQueue::triangulateMonotoneBlock_( EdgeId holeEdgeId )
{
    MR_TIMER;
    auto holeLoop = trackRightBoundaryLoop( tp_, holeEdgeId );
    auto lessPred = [&] ( EdgeId l, EdgeId r )
    {
        return less_( tp_.org( l ) , tp_.org( r ) );
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
        pvc[2].id = tp_.org( holeLoop[cur] );
        pvc[0].id = tp_.org( holeLoop[prev] );
        pvc[1].id = tp_.org( holeLoop[next] );
        for ( int i = 0; i < 3; ++i )
            pvc[i].pt = to2dim( pts_[pvc[i].id] );
        return ccw( pvc ) == lowerChain;
    };

    auto addDiagonal = [&] ( int cur, int prev, bool lowerChain )->bool
    {
        auto& tp = tp_;
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
        //assert( lessPred( holeLoop[curLower], holeLoop[nextLower] ) );
        //assert( lessPred( holeLoop[curUpper], holeLoop[nextUpper] ) );
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

Mesh getOutlineMesh( const Contours2d& contsd, IntersectionsMap* interMap /*= nullptr */, const BaseOutlineParameters& params )
{
    SweepLineQueue triangulator( contsd, nullptr, false, params.innerType, true, params.allowMerge );

    if ( interMap )
        interMap->shift = triangulator.vertSize();
    auto mesh = triangulator.run( interMap );
    if ( !mesh )
    {
        assert( false );
        return {};
    }
    return *mesh;
}

Mesh getOutlineMesh( const Contours2f& contours, IntersectionsMap* interMap /*= nullptr */, const BaseOutlineParameters& params )
{
    const auto contsd = copyContours<Contours2d>( contours );
    return getOutlineMesh( contsd, interMap, params );
}

Contours2f getOutline( const Contours2d& contours, const OutlineParameters& params )
{
    IntersectionsMap interMap;
    auto mesh = getOutlineMesh( contours, params.indicesMap ? &interMap : nullptr, params.baseParams );

    // `getValidFaces` important to exclude lone boundaries
    auto bourndaries = findRightBoundary( mesh.topology, &mesh.topology.getValidFaces() );
    Contours2f res;
    res.reserve( bourndaries.size() );
    for ( int i = 0; i < bourndaries.size(); ++i )
    {
        const auto& loop = bourndaries[i];
        res.push_back( {} );
        res.back().reserve( loop.size() + 1 );
        if ( params.indicesMap )
        {
            params.indicesMap->push_back( {} );
            params.indicesMap->back().reserve( loop.size() + 1 );
        }

        for ( auto e : loop )
        {
            VertId v = mesh.topology.org( e );
            res.back().push_back( to2dim( mesh.points[v] ) );
            if ( params.indicesMap )
            {
                if ( v < interMap.shift )
                    params.indicesMap->back().push_back( { .lOrg = v } );
                else
                {
                    const auto& inter = interMap.map[int( v ) - interMap.shift];
                    params.indicesMap->back().push_back( inter );
                }
            }
        }
        res.back().push_back( to2dim( mesh.destPnt( loop.back() ) ) );
        if ( params.indicesMap )
            params.indicesMap->back().push_back( params.indicesMap->back().front() );
    }
    return res;
}

Contours2f getOutline( const Contours2f& contours, const OutlineParameters& params )
{
    const auto contsd = copyContours<Contours2d>( contours );
    return getOutline( contsd, params );
}

Mesh triangulateContours( const Contours2d& contours, const HolesVertIds* holeVertsIds /*= nullptr*/ )
{
    if ( contours.empty() )
        return {};
    SweepLineQueue triangulator( contours, holeVertsIds, false, WindingMode::NonZero );
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
    return triangulateContours( contsd, holeVertsIds );
}

std::optional<Mesh> triangulateDisjointContours( const Contours2d& contours, const HolesVertIds* holeVertsIds /*= nullptr*/, std::vector<EdgePath>* outBoundaries /*= nullptr*/ )
{
    if ( contours.empty() )
        return Mesh();
    SweepLineQueue triangulator( contours, holeVertsIds, true, WindingMode::NonZero, false, true, outBoundaries );
    return triangulator.run();
}

std::optional<Mesh> triangulateDisjointContours( const Contours2f& contours, const HolesVertIds* holeVertsIds /*= nullptr*/, std::vector<EdgePath>* outBoundaries /*= nullptr*/ )
{
    const auto contsd = copyContours<Contours2d>( contours );
    return triangulateDisjointContours( contsd, holeVertsIds, outBoundaries );
}

}

TEST( MRMesh, PlanarTriangulation )
{
    // Create a quadrangle with three points on a straight line
    Contour2f cont;
    cont.push_back( Vector2f( 1.f, 0.f ) );
    cont.push_back( Vector2f( 0.f, 0.f ) );
    cont.push_back( Vector2f( 0.f, 1.f ) );
    cont.push_back( Vector2f( 0.f, 2.f ) );
    cont.push_back( Vector2f( 1.f, 0.f ) );

    Mesh mesh = PlanarTriangulation::triangulateContours( { cont } );
    mesh.pack();
    EXPECT_TRUE( mesh.topology.lastValidFace() == 1_f );
    // Must not contain degenerate faces
    EXPECT_TRUE( mesh.triangleAspectRatio( 0_f ) < 10.0f );
    EXPECT_TRUE( mesh.triangleAspectRatio( 1_f ) < 10.0f );
}

}
