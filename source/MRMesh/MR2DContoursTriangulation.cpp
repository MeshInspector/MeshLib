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
#include "MRPolyline.h"
#include "MRLinesSave.h"
#include "MRPointCloud.h"
#include "MRPointsSave.h"
#include "MRBitSetParallelFor.h"

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

class SweepLineQueue
{
public:
    SweepLineQueue( Mesh& mesh, const CoordinateConverters2& converters );
    std::vector<Vector3f> findIntersections();
private:
    Mesh& mesh_;
    const CoordinateConverters2& converters_;
    std::vector<Vector3f> intersections_;


    // sorted vertices with no left-going edges
    std::vector<ComaparableVertId> startVerts_;
    // index of next `startVerts_` 
    int startVertIndex_{ 0 };

    struct SweepEdgeInfo
    {
        EdgeId edgeId;
        VertId interWithLowerId;
        VertId interWithUpperId;
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
        // EventType::Start - index of candidate in `startVerts_`
        // EventType::Destination - id of lowest edge (with this destenation) in `activeSweepEdges_`
        // EventType::Intersection - id of lowest edge (with this intersection) in `activeSweepEdges_`
        int index{ -1 }; // -1 means that we finished queue
        // return true if event is valid
        operator bool() const { return index != -1; }
    };
    // get next queue element
    Event getNext_();

    void invalidateIntersection_( int indexLower );
    bool isIntersectionValid_( int indexLower );

    std::vector<SweepEdgeInfo> rightGoingCache_;
    std::vector<EdgeId> findClosestCache_;
    void processStartEvent_( int index );
    void processDestenationEvent_( int index );
    void processIntersectionEvent_( int index );

    using IntersectionMap = HashMap<std::pair<EdgeId, EdgeId>, VertId>;
    IntersectionMap intersectionsMap_; // needed to prevent recreation of same vertices multiple times
    void checkIntersection_( int index, bool lower );
    void checkIntersection_( int indexLower );
};

SweepLineQueue::SweepLineQueue( Mesh& mesh, const CoordinateConverters2& converters ):
    mesh_{ mesh},
    converters_{ converters }
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
    int i = 0;
    for ( auto v : startVertices )
        startVerts_[i++] = ComaparableVertId( &mesh_, v, &converters_ );

    std::sort( startVerts_.begin(), startVerts_.end() );
}

std::vector<Vector3f> SweepLineQueue::findIntersections()
{
    while ( auto event = getNext_() )
    {
        if ( event.type == EventType::Start )
            processStartEvent_( event.index );
        else if ( event.type == EventType::Destination )
            processDestenationEvent_( event.index );
        else if ( event.type == EventType::Intersection )
            processIntersectionEvent_( event.index );
    }
    return std::move( intersections_ );
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
        if ( !activeSweep.interWithUpperId )
            continue;
        auto candidate = ComaparableVertId( &mesh_, activeSweep.interWithUpperId, &converters_ );
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
        }
    }

    if ( startVertIndex_ < startVerts_.size() )
    {
        if ( !minDest.id || startVerts_[startVertIndex_] < minDest )
        {
            outEvent.type = EventType::Start;
            outEvent.index = startVertIndex_;
            ++startVertIndex_;
        }
    }
    return outEvent;
}

void SweepLineQueue::invalidateIntersection_( int indexLower )
{
    if ( indexLower >= 0 && indexLower < activeSweepEdges_.size() )
        activeSweepEdges_[indexLower].interWithUpperId = {};
    if ( indexLower + 1 >= 0 && indexLower + 1 < activeSweepEdges_.size() )
        activeSweepEdges_[indexLower + 1].interWithLowerId = {};
}

bool SweepLineQueue::isIntersectionValid_( int indexLower )
{
    if ( indexLower < 0 || indexLower + 1 >= activeSweepEdges_.size() )
        return false;
    if ( !activeSweepEdges_[indexLower].interWithUpperId )
        return false;
    return activeSweepEdges_[indexLower].interWithUpperId == activeSweepEdges_[indexLower + 1].interWithLowerId;
}

void SweepLineQueue::processStartEvent_( int index )
{
    int activeVPosition{ INT_MAX };// index of first edge, under activeV (INT_MAX - all edges are lower, -1 - all edges are upper)
    std::array<PreciseVertCoords2, 3> pvc;
    pvc[1].id = startVerts_[index].id;
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

    rightGoingCache_.clear();
    findClosestCache_.clear();
    findClosestCache_.emplace_back( EdgeId{} );
    for ( auto e : orgRing( mesh_.topology, startVerts_[index].id ) )
    {
        rightGoingCache_.emplace_back( SweepEdgeInfo{ .edgeId = e } );
        findClosestCache_.push_back( e );
    }

    auto lowestRight = findClosestToFront( mesh_, converters_, findClosestCache_, true ) - 1;
    assert( lowestRight > -1 );
    
    std::rotate( rightGoingCache_.begin(), rightGoingCache_.begin() + lowestRight, rightGoingCache_.end() );

    auto pos = activeVPosition == INT_MAX ? int( activeSweepEdges_.size() ) : activeVPosition + 1;
    
    invalidateIntersection_( pos - 1 );

    activeSweepEdges_.insert( activeSweepEdges_.begin() + pos, rightGoingCache_.begin(), rightGoingCache_.end() );

    checkIntersection_( pos, true );
    checkIntersection_( pos + 1, false );
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
    if ( numRight == 0 )
    {
        activeSweepEdges_.erase( activeSweepEdges_.begin() + minIndex, activeSweepEdges_.begin() + maxIndex + 1 );
        checkIntersection_( minIndex - 1, false );
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

        checkIntersection_( minIndex + numRight, true );
        checkIntersection_( minIndex - 1, false );
    }
}

void SweepLineQueue::processIntersectionEvent_( int index )
{
    bool isValid = isIntersectionValid_( index );
    if ( isValid )
        intersections_.push_back( mesh_.points[activeSweepEdges_[index].interWithUpperId] );
    invalidateIntersection_( index );
    if ( !isValid )
        return;

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
    auto& vertId = intersectionsMap_[{minEdgeId, maxEdgeId}];
    if ( !vertId )
    {
        vertId = mesh_.addPoint( to3dim(
            findSegmentSegmentIntersectionPrecise(
                to2dim( mesh_.points[pvc[0].id] ), to2dim( mesh_.points[pvc[1].id] ),
                to2dim( mesh_.points[pvc[2].id] ), to2dim( mesh_.points[pvc[3].id] ),
                converters_ ) ) );
    }

    activeSweepEdges_[i].interWithUpperId = vertId;
    activeSweepEdges_[i + 1].interWithLowerId = vertId;
}

// class for first sweep line pass for resolving intersections
class Intersector
{
public:
    Intersector( Mesh& mesh, const CoordinateConverters2& convertes );
    std::vector<Vector3f> findIntersections();
private:
    Mesh& mesh_;
    const CoordinateConverters2& converters_;

    struct Event : ComaparableVertId
    {
        EdgeId lower;
        EdgeId upper;
    };
    using InterQueue = std::priority_queue<Event, std::vector<Event>, std::greater<Event>>;
    InterQueue interQueue_;

    using ValidationMap = HashMap<std::pair<EdgeId, EdgeId>, bool>;
    ValidationMap validationMap_;

    void invalidateIntersection_( EdgeId a, EdgeId b );
    bool validateIntersection_( EdgeId a, EdgeId b );
    bool isIntersectionValid_( EdgeId a, EdgeId b );

    std::vector<Vector3f> intersections_;
    void processEvent_( const Event& event );
    void checkIntersection_( int index, bool lower );
    void checkIntersection_( int indexLower );
    std::vector<EdgeId> sweepInfo_;
};

Intersector::Intersector( Mesh& mesh, const CoordinateConverters2& convertes ) :
    mesh_{ mesh }, converters_{ convertes }
{
    for ( auto v : mesh.topology.getValidVerts() )
        interQueue_.push( Event{ ComaparableVertId( &mesh,v,&converters_ ) } );
}

std::vector<Vector3f> Intersector::findIntersections()
{
    MR_TIMER;
    while ( !interQueue_.empty() )
    {
        auto event = interQueue_.top();
        interQueue_.pop();
        processEvent_( event );
    }
    return std::move( intersections_ );
}

void Intersector::invalidateIntersection_( EdgeId a, EdgeId b )
{
    if ( b < a )
        std::swap( a, b );
    auto it = validationMap_.find( { a,b } );
    if ( it == validationMap_.end() )
        return;
    it->second = false;
}

bool Intersector::validateIntersection_( EdgeId a, EdgeId b )
{
    if ( b < a )
        std::swap( a, b );
    auto [it, inserted] = validationMap_.insert_or_assign( { a,b }, true );
    return inserted;
}

bool Intersector::isIntersectionValid_( EdgeId a, EdgeId b )
{
    if ( b < a )
        std::swap( a, b );
    auto it = validationMap_.find( { a,b } );
    if ( it == validationMap_.end() )
        return false;
    return it->second;
}

void Intersector::processEvent_( const Event& event )
{
    if ( event.lower && event.upper ) // intersections mode
    {
        if ( !isIntersectionValid_( event.lower, event.upper ) )
            return;

        auto it = std::find( sweepInfo_.begin(), sweepInfo_.end(), event.lower );
        if ( it == sweepInfo_.end() )
            return; // this intersection was already processed
        auto pos = int( std::distance( sweepInfo_.begin(), it ) );
        if ( pos + 1 == sweepInfo_.size() )
            return; // this intersection was already processed
        if ( sweepInfo_[pos + 1] != event.upper )
            return; // this intersection was already processed

        if ( pos > 0 )
            invalidateIntersection_( sweepInfo_[pos - 1], sweepInfo_[pos] );
        if ( pos + 2 < sweepInfo_.size() )
            invalidateIntersection_( sweepInfo_[pos + 1], sweepInfo_[pos + 2] );

        intersections_.push_back( mesh_.points[event.id] );

        std::swap( sweepInfo_[pos], sweepInfo_[pos + 1] );
        checkIntersection_( pos, true );
        checkIntersection_( pos + 1, false );
        return;
    }
    std::array<EdgeId, 2> rightEdges;
    std::array<int, 2> indicesToRemove = { -1,-1 };
    for ( auto e : orgRing( mesh_.topology, event.id ) )
    {
        auto it = std::find_if( sweepInfo_.begin(), sweepInfo_.end(), [&] ( EdgeId sE )
        {
            return e.sym() == sE;
        } );
        if ( it == sweepInfo_.end() )
        {
            if ( !rightEdges[0] )
                rightEdges[0] = e;
            else
            {
                // ATTENTION: merge same points cause this assert to strike
                assert( !rightEdges[1] );
                rightEdges[1] = e;
            }
        }
        else
        {
            if ( indicesToRemove[0] == -1 )
                indicesToRemove[0] = int( std::distance( sweepInfo_.begin(), it ) );
            else
            {
                // ATTENTION: merge same points cause this assert to strike
                assert( indicesToRemove[1] == -1 );
                indicesToRemove[1] = int( std::distance( sweepInfo_.begin(), it ) );
            }
        }
    }

    if ( rightEdges[0] && indicesToRemove[0] >= 0 )
    {
        assert( !rightEdges[1] );
        assert( indicesToRemove[1] == -1 );
        sweepInfo_[indicesToRemove[0]] = rightEdges[0];
        checkIntersection_( indicesToRemove[0], true );
        checkIntersection_( indicesToRemove[0], false );
        return;
    }

    if ( indicesToRemove[1] >= 0 )
    {
        assert( indicesToRemove[0] >= 0 );
        assert( !rightEdges[0] );
        assert( !rightEdges[1] );
        if ( indicesToRemove[0] > indicesToRemove[1] )
            std::swap( indicesToRemove[0], indicesToRemove[1] );
        // !!! this can strike sometimes
        // intersection is later in queue than this vertex
        assert( indicesToRemove[1] - indicesToRemove[0] == 1 );

        // remove left
        for ( int i = 1; i >= 0; --i )
            sweepInfo_.erase( sweepInfo_.begin() + indicesToRemove[i] );

        checkIntersection_( indicesToRemove[0] - 1, false );
        return;
    }

    assert( rightEdges[0] );
    assert( rightEdges[1] );
    assert( indicesToRemove[0] == -1 );
    assert( indicesToRemove[1] == -1 );

    int activeVPosition{ INT_MAX };// index of first edge, under activeV (INT_MAX - all edges are lower, -1 - all edges are upper)
    std::array<PreciseVertCoords2, 3> pvc;
    pvc[1].id = event.id;
    pvc[1].pt = converters_.toInt( to2dim( mesh_.points[pvc[1].id] ) );
    for ( int i = 0; i < sweepInfo_.size(); ++i )
    {
        pvc[0].id = mesh_.topology.org( sweepInfo_[i] );
        pvc[2].id = mesh_.topology.dest( sweepInfo_[i] );
        pvc[0].pt = converters_.toInt( to2dim( mesh_.points[pvc[0].id] ) );
        pvc[2].pt = converters_.toInt( to2dim( mesh_.points[pvc[2].id] ) );

        if ( activeVPosition == INT_MAX && ccw( pvc ) )
            activeVPosition = i - 1;
    }

    pvc[0] = pvc[1];
    pvc[1].id = mesh_.topology.dest( rightEdges[0] );
    pvc[2].id = mesh_.topology.dest( rightEdges[1] );
    for ( int i = 1; i < 3; ++i )
        pvc[i].pt = converters_.toInt( to2dim( mesh_.points[pvc[i].id] ) );

    if ( !ccw( pvc ) )
        std::swap( rightEdges[0], rightEdges[1] );

    auto pos = activeVPosition == INT_MAX ? int( sweepInfo_.size() ) : activeVPosition + 1;

    if ( pos > 0 && pos < sweepInfo_.size() )
        invalidateIntersection_( sweepInfo_[pos - 1], sweepInfo_[pos] );

    sweepInfo_.insert( sweepInfo_.begin() + pos, rightEdges.begin(), rightEdges.end() );

    checkIntersection_( pos, true );
    checkIntersection_( pos + 1, false );
}

void Intersector::checkIntersection_( int index, bool lower )
{
    if ( index < 0 || index >= sweepInfo_.size() )
        return;
    if ( lower && index == 0 )
        return;
    if ( !lower && index + 1 == sweepInfo_.size() )
        return;
    if ( lower && index >= 1 )
        return checkIntersection_( index - 1 );
    if ( !lower && index +1 < sweepInfo_.size() )
        return checkIntersection_( index );
}

void Intersector::checkIntersection_( int i )
{
    assert( i >= 0 && i + 1 < sweepInfo_.size() );

    // fill up
    std::array<PreciseVertCoords2, 4> pvc;
    auto org1 = mesh_.topology.org( sweepInfo_[i] );
    auto dest1 = mesh_.topology.dest( sweepInfo_[i] );
    auto org2 = mesh_.topology.org( sweepInfo_[i + 1] );
    auto dest2 = mesh_.topology.dest( sweepInfo_[i + 1] );
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

    if ( !validateIntersection_( sweepInfo_[i], sweepInfo_[i + 1] ) )
        return; // just validate is enough

    auto vertId = mesh_.addPoint( to3dim(
        findSegmentSegmentIntersectionPrecise(
            to2dim( mesh_.points[pvc[0].id] ), to2dim( mesh_.points[pvc[1].id] ),
            to2dim( mesh_.points[pvc[2].id] ), to2dim( mesh_.points[pvc[3].id] ),
            converters_ ) ) );

    Event event{ ComaparableVertId{&mesh_,vertId,&converters_} };
    event.lower = sweepInfo_[i];
    event.upper = sweepInfo_[i + 1];
    interQueue_.push( std::move( event ) );
}

class PlanarTriangulator
{
public:
    // constructor makes initial mesh which simply contain input contours as edges
    // if holesVertId is null - merge all vertices with same coordinates
    // otherwise only merge the ones with same initial vertId
    PlanarTriangulator( const Contours2d& contours, const HolesVertIds* holesVertId = nullptr, 
        bool abortWhenIntersect = false, WindingMode mode = WindingMode::NonZero );
    // process line sweep queue and triangulate inside area of mesh (based on winding rule)
    std::optional<Mesh> run();
    void saveDegubPoints();
private:
    Mesh mesh_;
    bool abortWhenIntersect_ = false;
    // this flag is set true if triangulation requires merging of two points that is forbidden
    bool incompleteMerge_ = false;
    WindingMode windingMode_{ WindingMode::NonZero };


    struct EdgeWindingInfo
    {
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

    CoordinateConverters2 converters_;

    std::priority_queue<ComaparableVertId, std::vector<ComaparableVertId>, std::greater<ComaparableVertId>> queue_;

    ComaparableVertId createCompVert_( VertId v );

    std::vector<EdgeId> findClosestCache_;
    int findClosestToFront_( const std::vector<EdgeId>& edges, bool right );

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
        LoneRightmostLeft loneRightmostLeft; // represents lone left upwards
    };
    std::vector<ActiveEdgeInfo> activeSweepEdges_;
    bool processOneVert_( VertId v );
    bool resolveIntersectios_();
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

PlanarTriangulator::PlanarTriangulator( const Contours2d& contours, const HolesVertIds* holesVertId /*= true*/, 
    bool abortWhenIntersect /*= false*/, WindingMode mode /*= WindingMode::NonZero*/ )
{
    windingMode_ = mode;
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
        if ( !windingInfo_[e].inside( windingMode_ ) )
            continue;
    
        triangulateMonotoneBlock_( dirE );
    }
    makeDeloneEdgeFlips( mesh_, {}, 300 );

    return std::move( mesh_ ); // move here to avoid copy of class member
}

void PlanarTriangulator::saveDegubPoints()
{
    auto meshCpy = mesh_;
    SweepLineQueue swQueue( mesh_, converters_ );
    PointCloud pc;
    pc.points.vec_ = swQueue.findIntersections();
    pc.validPoints.resize( pc.points.vec_.size() );
    pc.validPoints.set();
    PointsSave::toCtm( pc, "C:\\Users\\grant\\Downloads\\terrain (1)\\intersections.ctm" );
}

ComaparableVertId PlanarTriangulator::createCompVert_( VertId v )
{
    return ComaparableVertId( &mesh_, v, &converters_ );
}

int PlanarTriangulator::findClosestToFront_( const std::vector<EdgeId>& edges, bool left )
{
    return findClosestToFront( mesh_, converters_, edges, left );
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
        bool sameIntCoord = createCompVert_( sortedPoints[i].id ) == createCompVert_( sortedPoints[prevUnique].id );
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

void PlanarTriangulator::calculateWinding_()
{
    int windingLast = 0;
    // recalculate winding number for active edges
    for ( const auto& e : activeSweepEdges_ )
    {
        if ( mesh_.topology.isLoneEdge( e.id ) )
            continue;
        auto& info = windingInfo_[e.id.undirected()];
        if ( info.windingModifier != INT_MAX )
            info.winding = windingLast + info.windingModifier;
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
    bool hasLeft = false;
    std::vector<ActiveEdgeInfo> rightGoingEdges;
    // information about removed lone left
    LoneRightmostLeft removedLoneLeftInfo; 
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
            hasLeft = true;
        }
    }
    EdgeId lowestLeftEdge;
    if ( hasLeft )
    {
        // also remove lone left after fixing duplicates
        for ( int i = 0; i < activeSweepEdges_.size(); ++i )
            if ( mesh_.topology.isLoneEdge( activeSweepEdges_[i].id ) )
                indicesToRemoveFromActive.push_back( i );
        // find lowest left for helper
        std::sort( indicesToRemoveFromActive.begin(), indicesToRemoveFromActive.end() );
        lowestLeftEdge = activeSweepEdges_[indicesToRemoveFromActive[0]].id.sym();

        // save info about removed lone
        for ( auto index : indicesToRemoveFromActive )
        {
            if ( !activeSweepEdges_[index].loneRightmostLeft.id )
                continue;
            assert( !removedLoneLeftInfo.id || removedLoneLeftInfo.id == activeSweepEdges_[index].loneRightmostLeft.id );
            removedLoneLeftInfo = activeSweepEdges_[index].loneRightmostLeft;
        }

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
    assert( hasLeft || !rightGoingEdges.empty() );

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
    if ( !hasLeft && hasOuter && windingInfo_[activeSweepEdges_[activeVPosition].id.undirected()].inside( windingMode_ ) )
    {
        assert( lowestRight != INT_MAX );
        // find helper:
        // id of rightmost left vertex (it's lower edge) closest to active vertex
        // close to `helper` described here : https://www.cs.umd.edu/class/spring2020/cmsc754/Lects/lect05-triangulate.pdf
        EdgeId helperId;
        const auto& lower = activeSweepEdges_[activeVPosition];
        const auto& upper = activeSweepEdges_[activeVPosition + 1];
        auto& loneLeft = activeSweepEdges_[activeVPosition].loneRightmostLeft;
        if ( loneLeft.id && loneLeft.lower == lower.id && loneLeft.upper == upper.id )
        {
            helperId = loneLeft.id;
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
        if ( helperId == loneLeft.id )
            loneLeft.id = EdgeId{};
        auto newE = mesh_.topology.makeEdge();
        mesh_.topology.splice( helperId, newE );
        mesh_.topology.splice( mesh_.topology.prev( rightGoingEdges[lowestRight].id ), newE.sym() );
        windingInfo_.resize( newE.undirected() + 1 );
        windingInfo_[newE.undirected()].winding = windingInfo_[lower.id.undirected()].winding;
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
            windingInfo_[newE.undirected()].winding = windingInfo_[loneInfo.lower.undirected()].winding;
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
        if ( removedLoneLeftInfo.id )
            connect( removedLoneLeftInfo );
    }

    // insert right going to active
    if ( !rightGoingEdges.empty() )
    {
        std::rotate( rightGoingEdges.begin(), rightGoingEdges.begin() + lowestRight, rightGoingEdges.end() );
        auto pos = activeVPosition == INT_MAX ? int( activeSweepEdges_.size() ) : activeVPosition + 1;
        activeSweepEdges_.insert( activeSweepEdges_.begin() + pos, rightGoingEdges.begin(), rightGoingEdges.end() );
    }
    else if ( hasOuter && windingInfo_[activeSweepEdges_[activeVPosition].id.undirected()].inside( windingMode_ ) )
    {
        assert( hasLeft );
        LoneRightmostLeft loneRightmostLeft;
        loneRightmostLeft.id = lowestLeftEdge;
        loneRightmostLeft.lower = activeSweepEdges_[activeVPosition].id;
        loneRightmostLeft.upper = activeSweepEdges_[activeVPosition + 1].id;

        activeSweepEdges_[activeVPosition].loneRightmostLeft = loneRightmostLeft;
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
        
        if ( !ccw( { pvc[1],pvc[3],interPvc } ) )
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
        windingInfo_[e1n.undirected()].windingModifier = windingInfo_[activeSweepEdges_[i].id.undirected()].windingModifier;
        windingInfo_[e2n.undirected()].windingModifier = windingInfo_[activeSweepEdges_[i + 1].id.undirected()].windingModifier;
        
        // we should really check for merge origins also, 
        // but it will require to precess origin one more time and break structure stability
        if ( interPvc.pt == pvc[1].pt )
            mergeSinglePare_( dest1, interPvc.id );
        else if ( interPvc.pt == pvc[3].pt )
            mergeSinglePare_( dest2, interPvc.id );
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

Mesh triangulateContours( const Contours2d& contours, const HolesVertIds* holeVertsIds /*= nullptr*/,
    WindingMode mode )
{
    PlanarTriangulator triangulator( contours, holeVertsIds, false, mode );
    triangulator.saveDegubPoints();
    auto res = triangulator.run();
    assert( res );
    if ( res )
        return std::move( *res );
    else
        return Mesh();
}

Mesh triangulateContours( const Contours2f& contours, const HolesVertIds* holeVertsIds /*= nullptr*/,
    WindingMode mode )
{
    const auto contsd = copyContours<Contours2d>( contours );
    return triangulateContours( contsd, holeVertsIds, mode );
}

std::optional<Mesh> triangulateDisjointContours( const Contours2d& contours, const HolesVertIds* holeVertsIds /*= nullptr*/ )
{
    PlanarTriangulator triangulator( contours, holeVertsIds, true );
    return triangulator.run();
}

std::optional<Mesh> triangulateDisjointContours( const Contours2f& contours, const HolesVertIds* holeVertsIds /*= nullptr*/ )
{
    const auto contsd = copyContours<Contours2d>( contours );
    return triangulateDisjointContours( contsd, holeVertsIds );
}

}

}