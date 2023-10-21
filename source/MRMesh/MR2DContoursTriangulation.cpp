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
#include <queue>
#include <algorithm>
#include <limits>

namespace MR
{

namespace PlanarTriangulation
{

int findClosestToFront( const Mesh& mesh, const CoordinateConverters2& converters,
    const std::vector<EdgeId>& edges, bool left )
{
    if ( edges.size() == 2 )
        return 1;
    std::array<PreciseVertCoords2, 3> pvc;
    auto org = mesh.topology.org( edges[1] );
    pvc[2].id = org;
    pvc[2].pt = converters.toInt( to2dim( mesh.points[org] ) );
    PreciseVertCoords2 baseVertCoord;
    if ( edges[0] )
    {
        auto dest = mesh.topology.dest( edges[0] );
        for ( int i = 1; i < edges.size(); ++i )
        {
            if ( dest == mesh.topology.dest( edges[i] ) )
                return i;
        }
        baseVertCoord.id = dest;
        baseVertCoord.pt = converters.toInt( to2dim( mesh.points[dest] ) );
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

        auto dest = mesh.topology.dest( edges[i] );
        pvc[1].id = dest;
        pvc[1].pt = converters.toInt( to2dim( mesh.points[dest] ) );
        PreciseVertCoords2 coordI = pvc[1];

        bool ccwBI = ccw( pvc );
        int nextI = getNextI( i, ccwBI );

        dest = mesh.topology.dest( edges[nextI] );
        pvc[1].id = dest;
        pvc[1].pt = converters.toInt( to2dim( mesh.points[dest] ) );

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

// struct to use easily compare mesh points by sweep line compare rule
struct ComaparableVertId
{
    ComaparableVertId() = default;
    ComaparableVertId( const Mesh* meshP, VertId v, const CoordinateConverters2* conv ) :
        mesh{ meshP }, id{ v }, converters{ conv }
    {}
    const Mesh* mesh{ nullptr };
    VertId id;
    const CoordinateConverters2* converters{ nullptr };
    bool operator<( const ComaparableVertId& other ) const;
    bool operator>( const ComaparableVertId& other ) const;
    bool operator==( const ComaparableVertId& other ) const;
};

bool ComaparableVertId::operator<( const ComaparableVertId& other ) const
{
    auto l = converters->toInt( to2dim( mesh->points[id] ) );
    auto r = converters->toInt( to2dim( other.mesh->points[other.id] ) );
    return l.x < r.x || ( l.x == r.x && l.y < r.y ) || ( l.x == r.x && l.y == r.y && id < other.id );
}

bool ComaparableVertId::operator>( const ComaparableVertId& other ) const
{
    auto l = converters->toInt( to2dim( mesh->points[id] ) );
    auto r = converters->toInt( to2dim( other.mesh->points[other.id] ) );
    return l.x > r.x || ( l.x == r.x && l.y > r.y ) || ( l.x == r.x && l.y == r.y && id > other.id );
}

bool ComaparableVertId::operator==( const ComaparableVertId& other ) const
{
    auto l = converters->toInt( to2dim( mesh->points[id] ) );
    auto r = converters->toInt( to2dim( other.mesh->points[other.id] ) );
    return l == r;
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
        bool needOutline = false // if set do not do real triangulation, just marks inside faces as present
        );

    std::optional<Mesh> run();

    bool findIntersections();
    void injectIntersections();
    void makeMonotone();
    void triangulate();
private:
    Mesh mesh_;
    CoordinateConverters2 converters_;

// INITIALIZATION CLASS BLOCK
    // if set only marks inside faces as present (for further findng outline)
    bool needOutline_ = false;
    // if set fails on first found intersection
    bool abortWhenIntersect_ = false;
    // this flag is set true if triangulation requires merging of two points that is forbidden
    bool incompleteMerge_ = false;
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
    std::vector<ComaparableVertId> startVerts_;
    std::vector<EdgeId> startVertLowestRight_;
    // index of next `startVerts_` 
    int startVertIndex_{ 0 };

    struct SweepEdgeInfo
    {
        EdgeId edgeId;
        union Info
        {
            VertId interVertId;
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
    using IntersectionMap = HashMap<std::pair<EdgeId, EdgeId>, IntersectionInfo>;
    IntersectionMap intersectionsMap_; // needed to prevent recreation of same vertices multiple times
    void checkIntersection_( int index, bool lower );
    void checkIntersection_( int indexLower );
};

SweepLineQueue::SweepLineQueue( 
    const Contours2d& contours, 
    const HolesVertIds* holesVertId,
    bool abortWhenIntersect, 
    WindingMode mode,
    bool needOutline ) :
    needOutline_{ needOutline },
    abortWhenIntersect_{ abortWhenIntersect },
    windingMode_{ mode }
{
    initMeshByContours_( contours );

    auto box = Box3d( mesh_.computeBoundingBox() );
    converters_.toInt = [conv = getToIntConverter( box )] ( const Vector2f& coord )
    {
        return to2dim( conv( to3dim( coord ) ) );
    };
    converters_.toFloat = [conv = getToFloatConverter( box )] ( const Vector2i& coord )
    {
        return to2dim( conv( to3dim( coord ) ) );
    };

    mergeSamePoints_( holesVertId );

    setupStartVertices_();
}

std::optional<MR::Mesh> SweepLineQueue::run()
{
    MR_TIMER;
    if ( incompleteMerge_ )
        return {};
    if ( !findIntersections() )
        return {};
    injectIntersections();
    makeMonotone();
    triangulate();
    return std::move( mesh_ );
}

bool SweepLineQueue::findIntersections()
{
    MR_TIMER;
    stage_ = Stage::Intersections;
    events_.clear();
    events_.reserve( mesh_.topology.numValidVerts() * 2 );
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

void SweepLineQueue::injectIntersections()
{
    MR_TIMER;
    windingInfo_.resize( windingInfo_.size() + intersections_.size() * 2 );
    Vector<EdgeId, UndirectedEdgeId> oldToFirstNewEdgeMap( mesh_.topology.undirectedEdgeSize() );
    for ( const auto& inter : intersections_ )
    {
        // split edges
        // set new edge ids to the left and save old to the right
        // because of intersections order

        // prev lower
        auto pl = mesh_.topology.prev( inter.lower );
        // lower left
        auto ll = mesh_.topology.makeEdge();
        if ( inter.lower.odd() )
            ll = ll.sym(); // oddity should stay the same (for winding number)
        mesh_.topology.splice( pl, inter.lower );
        mesh_.topology.splice( pl, ll );
        mesh_.topology.splice( inter.lower, ll.sym() );

        // prev upper
        auto pu = mesh_.topology.prev( inter.upper );
        // upper left
        auto ul = mesh_.topology.makeEdge();
        if ( inter.upper.odd() )
            ul = ul.sym(); // oddity should stay the same (for winding number)

        mesh_.topology.splice( pu, inter.upper );
        mesh_.topology.splice( pu, ul );

        mesh_.topology.splice( inter.lower, ul.sym() );
        mesh_.topology.splice( ll.sym(), inter.upper );

        mesh_.topology.setOrg( inter.upper, inter.vId );

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
    for ( auto event : events_ )
    {
        if ( event.type == EventType::Start )
            processStartEvent_( event.index );
        else
            processDestenationEvent_( event.index );
        calculateWinding_();
    }
}

void SweepLineQueue::triangulate()
{
    MR_TIMER;
    stage_ = Stage::Triangulation;
    if ( !needOutline_ )
        reflexChainCache_.reserve( 256 ); // reserve once to have less allocations later
    for ( auto e : undirectedEdges( mesh_.topology ) )
    {
        if ( e >= windingInfo_.size() )
            continue;
        const auto& windInfo = windingInfo_[e];
        if ( !windInfo.inside( windingMode_ ) )
            continue;
        auto dirE = EdgeId( e << 1 );
        if ( !windInfo.rightGoing )
            dirE = dirE.sym();
        if ( mesh_.topology.left( dirE ) )
            continue;

        if ( !needOutline_ )
            triangulateMonotoneBlock_( dirE ); // triangulate
        else
            mesh_.topology.setLeft( dirE, mesh_.topology.addFaceId() ); // mark present
    }
    if ( !needOutline_ )
        makeDeloneEdgeFlips( mesh_, {}, 300 );
}

void SweepLineQueue::setupStartVertices_()
{
    VertBitSet startVertices( mesh_.topology.vertSize() );
    BitSetParallelFor( mesh_.topology.getValidVerts(), [&] ( VertId v )
    {
        auto thisVert = ComaparableVertId( &mesh_, v, &converters_ );
        bool startVert = true;
        for ( auto e : orgRing( mesh_.topology, v ) )
        {
            auto destComp = ComaparableVertId( &mesh_, mesh_.topology.dest( e ), &converters_ );
            if ( destComp < thisVert )
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
        startVerts_[i++] = ComaparableVertId( &mesh_, v, &converters_ );

    std::sort( startVerts_.begin(), startVerts_.end() );
}

SweepLineQueue::Event SweepLineQueue::getNext_()
{
    Event outEvent;
    int minInterIndex = -1;

    ComaparableVertId minDest( &mesh_, VertId{}, &converters_ );
    ComaparableVertId minInter( &mesh_, VertId{}, &converters_ );
    VertId prevDestId;
    for ( int i = 0; i < activeSweepEdges_.size(); ++i )
    {
        const auto& activeSweep = activeSweepEdges_[i];
        VertId destId = mesh_.topology.dest( activeSweep.edgeId );
        if ( destId != prevDestId )
        {
            prevDestId = destId;
            auto candidate = ComaparableVertId( &mesh_, destId, &converters_ );
            if ( !minDest.id || candidate < minDest )
            {
                minDest = candidate;
                outEvent.type = EventType::Destination;
                outEvent.index = i;
            }
        }
        if ( stage_ != Stage::Intersections || !activeSweep.upperInfo.interVertId )
            continue;
        auto candidate = ComaparableVertId( &mesh_, activeSweep.upperInfo.interVertId, &converters_ );
        if ( !minInter.id || candidate < minInter )
        {
            minInter = candidate;
            minInterIndex = i;
        }
    }

    if ( minInter.id )
    {
        if ( minInter < minDest ||
            mesh_.topology.dest( activeSweepEdges_[minInterIndex].edgeId ) == minDest.id || 
            mesh_.topology.dest( activeSweepEdges_[minInterIndex + 1].edgeId ) == minDest.id )
        {
            outEvent.type = EventType::Intersection;
            outEvent.index = minInterIndex;
            minDest = minInter;
        }
    }

    if ( startVertIndex_ < startVerts_.size() )
    {
        if ( !minDest.id || startVerts_[startVertIndex_] < minDest )
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
    pvc[1].id = startVerts_[startVertIndex_].id;
    pvc[1].pt = converters_.toInt( to2dim( mesh_.points[pvc[1].id] ) );
    for ( int i = 0; i < activeSweepEdges_.size(); ++i )
    {
        pvc[0].id = mesh_.topology.org( activeSweepEdges_[i].edgeId );
        pvc[2].id = mesh_.topology.dest( activeSweepEdges_[i].edgeId );
        pvc[0].pt = converters_.toInt( to2dim( mesh_.points[pvc[0].id] ) );
        pvc[2].pt = converters_.toInt( to2dim( mesh_.points[pvc[2].id] ) );

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
    for ( auto e : orgRing( mesh_.topology, startVerts_[startVertIndex_].id ) )
    {
        rightGoingCache_.emplace_back( SweepEdgeInfo{ .edgeId = e } );
        if ( stage_ == Stage::Intersections )
            findClosestCache_.push_back( e );
    }

    int pos = -1;
    if ( stage_ == Stage::Intersections )
    {
        pos = findClosestToFront( mesh_, converters_, findClosestCache_, true ) - 1;
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
            auto lowerOrg = ComaparableVertId( &mesh_, mesh_.topology.org( activeSweepEdges_[index - 1].edgeId ), &converters_ );
            auto upperOrg = ComaparableVertId( &mesh_, mesh_.topology.org( activeSweepEdges_[index].edgeId ), &converters_ );
            if ( lowerOrg < upperOrg )
                helperId = activeSweepEdges_[index - 1].edgeId;
            else
                helperId = mesh_.topology.prev( activeSweepEdges_[index].edgeId );
        }
        assert( helperId );

        auto newEdge = mesh_.topology.makeEdge();
        if ( activeSweepEdges_[index - 1].edgeId.odd() )
            newEdge = newEdge.sym();
        mesh_.topology.splice( helperId, newEdge );
        mesh_.topology.splice( rightGoingCache_.back().edgeId, newEdge.sym() );

        windingInfo_.autoResizeSet( newEdge.undirected(), windingInfo_[activeSweepEdges_[index - 1].edgeId.undirected()] );
    }

    activeSweepEdges_.insert( activeSweepEdges_.begin() + index, rightGoingCache_.begin(), rightGoingCache_.end() );

    if ( stage_ == Stage::Intersections )
    {
        checkIntersection_( index, true );
        checkIntersection_( index + 1, false );
    }

    ++startVertIndex_;
}

void SweepLineQueue::processDestenationEvent_( int index )
{
    int minIndex = index;
    int maxIndex = index;
    for ( int i = minIndex + 1; i < activeSweepEdges_.size(); ++i )
    {
        if ( mesh_.topology.dest( activeSweepEdges_[index].edgeId ) != mesh_.topology.dest( activeSweepEdges_[i].edgeId ) )
            break;
        maxIndex = i;
    }
    rightGoingCache_.clear();
    for ( auto e : orgRing0( mesh_.topology, activeSweepEdges_[minIndex].edgeId.sym() ) )
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
                connectorEdgeId = mesh_.topology.prev( activeSweepEdges_[i].edgeId.sym() );

            auto newEdge = mesh_.topology.makeEdge();
            if ( activeSweepEdges_[i].edgeId.odd() )
                newEdge = newEdge.sym();
            mesh_.topology.splice( lowerLone, newEdge );
            mesh_.topology.splice( connectorEdgeId, newEdge.sym() );

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
    auto org1 = mesh_.topology.org( activeSweepEdges_[i].edgeId );
    auto dest1 = mesh_.topology.dest( activeSweepEdges_[i].edgeId );
    auto org2 = mesh_.topology.org( activeSweepEdges_[i + 1].edgeId );
    auto dest2 = mesh_.topology.dest( activeSweepEdges_[i + 1].edgeId );
    bool canIntersect = org1 != org2 && dest1 != dest2;
    if ( !canIntersect || !org1 || !org2 || !dest1 || !dest2 )
        return;

    pvc[0].id = org1; pvc[1].id = dest1;
    pvc[2].id = org2; pvc[3].id = dest2;

    for ( int p = 0; p < 4; ++p )
        pvc[p].pt = converters_.toInt( to2dim( mesh_.points[pvc[p].id] ) );

    auto haveInter = doSegmentSegmentIntersect( pvc );
    if ( !haveInter.doIntersect )
        return;

    auto minEdgeId = std::min( activeSweepEdges_[i].edgeId, activeSweepEdges_[i + 1].edgeId );
    auto maxEdgeId = std::max( activeSweepEdges_[i].edgeId, activeSweepEdges_[i + 1].edgeId );
    auto& interInfo = intersectionsMap_[{minEdgeId, maxEdgeId}];
    if ( !interInfo )
    {
        interInfo.vId = mesh_.addPoint( to3dim(
            findSegmentSegmentIntersectionPrecise(
                to2dim( mesh_.points[pvc[0].id] ), to2dim( mesh_.points[pvc[1].id] ),
                to2dim( mesh_.points[pvc[2].id] ), to2dim( mesh_.points[pvc[3].id] ),
                converters_ ) ) );
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
    std::vector<ComaparableVertId> sortedPoints;
    sortedPoints.reserve( mesh_.points.size() );
    for ( int i = 0; i < mesh_.points.size(); ++i )
        sortedPoints.emplace_back( ComaparableVertId( &mesh_, VertId( i ), &converters_ ) );
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
        bool sameIntCoord = sortedPoints[i] == sortedPoints[prevUnique];
        if ( !sameIntCoord )
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
}

void SweepLineQueue::mergeSinglePare_( VertId unique, VertId same )
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
        auto minEUnique = findClosestCache_[findClosestToFront( mesh_, converters_, findClosestCache_, false )];
        auto prev = mesh_.topology.prev( eSame );
        if ( prev != eSame )
            mesh_.topology.splice( prev, eSame );
        else
            mesh_.topology.setOrg( eSame, VertId{} );
        mesh_.topology.splice( minEUnique, eSame );
        if ( mesh_.topology.dest( minEUnique ) == mesh_.topology.dest( eSame ) )
        {
            auto& edgeInfo = windingInfo_.autoResizeAt( minEUnique.undirected() );
            if ( edgeInfo.windingModifier == INT_MAX )
                edgeInfo.windingModifier = 1;
            bool uniqueIsOdd = minEUnique.odd();
            bool sameIsOdd = eSame.odd();
            edgeInfo.windingModifier += ( ( uniqueIsOdd == sameIsOdd ) ? 1 : -1 );
            mesh_.topology.splice( mesh_.topology.prev( eSame ), eSame );
            mesh_.topology.splice( mesh_.topology.prev( eSame.sym() ), eSame.sym() );
        }
    }
}

void SweepLineQueue::removeMultipleAfterMerge_()
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
        edgeInfo.windingModifier = 1;
        bool uniqueIsOdd = int( multiplesFromThis.front() ) & 1;
        for ( int i = 1; i < multiplesFromThis.size(); ++i )
        {
            auto e = multiplesFromThis[i];
            bool isMEOdd = int( e ) & 1;
            edgeInfo.windingModifier += ( ( uniqueIsOdd == isMEOdd ) ? 1 : -1 );
            mesh_.topology.splice( mesh_.topology.prev( e ), e );
            mesh_.topology.splice( mesh_.topology.prev( e.sym() ), e.sym() );
            assert( mesh_.topology.isLoneEdge( e ) );
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
    auto holeLoop = trackRightBoundaryLoop( mesh_.topology, holeEdgeId );
    auto lessPred = [&] ( EdgeId l, EdgeId r )
    {
        auto lComp = ComaparableVertId( &mesh_, mesh_.topology.org( l ), &converters_ );
        auto rComp = ComaparableVertId( &mesh_, mesh_.topology.org( r ), &converters_ );
        return lComp < rComp;
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

Mesh getOutlineMesh( const Contours2f& contours )
{
    const auto contsd = copyContours<Contours2d>( contours );
    SweepLineQueue triangulator( contsd, nullptr, false, WindingMode::Negative );
    auto mesh = triangulator.run();
    if ( !mesh )
    {
        assert( false );
        return {};
    }
    return *mesh;
}

Contours2f getOutline( const Contours2f& contours )
{
    auto mesh = getOutlineMesh( contours );

    // `getValidFaces` important to exclude lone boundaries
    auto bourndaries = findRightBoundary( mesh.topology, &mesh.topology.getValidFaces() );
    Contours2f res;
    res.reserve( bourndaries.size() );
    for ( int i = 0; i < bourndaries.size(); ++i )
    {
        const auto& loop = bourndaries[i];
        res.push_back( {} );
        res.back().reserve( loop.size() + 1 );

        for ( auto e : loop )
            res.back().push_back( to2dim( mesh.orgPnt( e ) ) );
        res.back().push_back( to2dim( mesh.destPnt( loop.back() ) ) );
    }
    return res;
}

Mesh triangulateContours( const Contours2d& contours, const HolesVertIds* holeVertsIds /*= nullptr*/ )
{
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

std::optional<Mesh> triangulateDisjointContours( const Contours2d& contours, const HolesVertIds* holeVertsIds /*= nullptr*/ )
{
    SweepLineQueue triangulator( contours, holeVertsIds, true );
    return triangulator.run();
}

std::optional<Mesh> triangulateDisjointContours( const Contours2f& contours, const HolesVertIds* holeVertsIds /*= nullptr*/ )
{
    const auto contsd = copyContours<Contours2d>( contours );
    return triangulateDisjointContours( contsd, holeVertsIds );
}

}

}