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
#include "MRPrecisePredicates3.h"
#include "MRPch/MRTBB.h"
#include <queue>
#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>

namespace MR
{

namespace PlanarTriangulation
{

// All coordinate-dependent operations of the sweep-line triangulation, factored out so a caller
// can supply alternative geometry. The default implementation is precisePredicates() below, which
// reproduces the historical exact-integer simulation-of-simplicity behavior.
struct SweepLinePredicates
{
    // strict order of vertices along the sweep direction (sweep advances toward the greater vertex)
    std::function<bool( VertId l, VertId r )> less;
    // strict lexicographic order of positions; used only to group coincident vertices before merging
    std::function<bool( VertId l, VertId r )> less2d;
    // whether two vertices share exactly the same position
    std::function<bool( VertId l, VertId r )> samePos;
    // orientation predicate, equivalent to MR::ccw( { a, b, c } )
    std::function<bool( VertId a, VertId b, VertId c )> ccw;
    // orientation of b around pivot relative to the direction the sweep arrives from
    // (i.e. ccw with the first point placed far behind pivot, opposite to the sweep)
    std::function<bool( VertId b, VertId pivot )> ccwFromBehind;
    // store the position of input vertex v, identified by its (contourId, pointId) in the source contours
    std::function<void( VertId v, int contourId, int pointId )> addInputPoint;
    // compute and store the position of intersection vertex v of segments (a,b) and (c,d)
    std::function<void( VertId v, VertId a, VertId b, VertId c, VertId d )> addIntersectionPoint;
    // position of vertex v in the output mesh
    std::function<Vector3f( VertId v )> point;
};

// the sweep-line predicates that depend only on the projected integer coordinates `pts2`;
// shared by precisePredicates (2D input) and meshSpacePredicates (3D mesh input projected on the dominant axis)
static void setPts2Predicates( SweepLinePredicates& p, std::shared_ptr<Vector<Vector2i, VertId>> pts2 )
{
    p.less = [pts2] ( VertId l, VertId r )
    {
        return smaller( { .id = l, .pt = ( *pts2 )[l].x }, { .id = r, .pt = ( *pts2 )[r].x } );
    };
    p.less2d = [pts2] ( VertId l, VertId r )
    {
        return std::tuple( ( *pts2 )[l].x, ( *pts2 )[l].y ) < std::tuple( ( *pts2 )[r].x, ( *pts2 )[r].y );
    };
    p.samePos = [pts2] ( VertId l, VertId r )
    {
        return ( *pts2 )[l] == ( *pts2 )[r];
    };
    p.ccw = [pts2] ( VertId a, VertId b, VertId c )
    {
        return MR::ccw( { PreciseVertCoords2{ a, ( *pts2 )[a] }, { b, ( *pts2 )[b] }, { c, ( *pts2 )[c] } } );
    };
    p.ccwFromBehind = [pts2] ( VertId b, VertId pivot )
    {
        Vector2i base = ( *pts2 )[pivot];
        base.x -= 10000; // far behind pivot along -X (the sweep axis), matching the historical reference
        return MR::ccw( { PreciseVertCoords2{ VertId{}, base }, { b, ( *pts2 )[b] }, { pivot, ( *pts2 )[pivot] } } );
    };
}

// default predicates: exact integer arithmetic with simulation-of-simplicity (historical behavior);
// `pts` is the storage for the projected points, cleared here; a cached buffer keeps its capacity
static SweepLinePredicates precisePredicates( const Contours2f& contours, std::shared_ptr<Vector<Vector2i, VertId>> pts )
{
    Box3f box;
    int pointsSize = 0;
    for ( const auto& cont : contours )
    {
        for ( const auto& p : cont )
            box.include( to3dim( p ) );
        if ( cont.size() > 3 )
        {
            assert( cont.front() == cont.back() );
            pointsSize += int( cont.size() ) - 1;
        }
    }

    pts->clear();
    pts->reserve( pointsSize );
    auto toInt = [conv = getToIntConverter( Box3d( box ) )] ( const Vector2f& coord )
    {
        return to2dim( conv( to3dim( coord ) ) );
    };
    auto toFloat = [conv = getToFloatConverter( Box3d( box ) )] ( const Vector2i& coord )
    {
        return to2dim( conv( to3dim( coord ) ) );
    };

    SweepLinePredicates p;
    setPts2Predicates( p, pts );
    // resolve an input vertex by (contourId, pointId); initMeshByContours_ drives the order, so
    // `contours` only needs to outlive construction (every caller passes its own input by reference)
    p.addInputPoint = [&contours, pts, toInt] ( VertId v, int contourId, int pointId )
    {
        pts->autoResizeSet( v, toInt( contours[contourId][pointId] ) );
    };
    p.addIntersectionPoint = [pts] ( VertId v, VertId a, VertId b, VertId c, VertId d )
    {
        pts->autoResizeSet( v, findSegmentSegmentIntersectionPrecise( ( *pts )[a], ( *pts )[b], ( *pts )[c], ( *pts )[d] ) );
    };
    p.point = [pts, toFloat] ( VertId v )
    {
        return to3dim( toFloat( ( *pts )[v] ) );
    };
    return p;
}

// per-contour vertex counts; lets the queue build the initial edge loops independently of coordinate dimension
static std::vector<int> getContourSizes( const Contours2f& contours )
{
    std::vector<int> sizes( contours.size() );
    for ( int i = 0; i < int( contours.size() ); ++i )
        sizes[i] = int( contours[i].size() );
    return sizes;
}

// map an input vertex (its index within the concatenated hole loops) back to its source mesh VertId;
// mirrors the walk in mergeSamePoints_
static VertId holeSourceVertId( const HolesVertIds& holes, VertId v )
{
    int idx = int( v );
    for ( const auto& hole : holes )
    {
        if ( idx < int( hole.size() ) )
            return hole[idx];
        idx -= int( hole.size() );
    }
    assert( false ); // a disjoint triangulation outputs only input vertices, all covered by holes
    return {};
}

// mesh-space predicates: triangulate hole boundary loops of `mesh` in the mesh's own 3D coordinates,
// orienting around `normal`. Combinatorics run on the dominant-axis projection (reusing the exact 2D
// predicates via setPts2Predicates). point() restores each output vertex's exact original mesh position
// through `holeVertIds` (no separate coordinate copy, no projection round-trip).
// `mesh`, `loops` and `holeVertIds` only need to outlive the run.
static SweepLinePredicates meshSpacePredicates( const Mesh& mesh, const EdgeLoops& loops, const Vector3f& normal, const HolesVertIds& holeVertIds,
    std::shared_ptr<Vector<Vector2i, VertId>> pts2 ) // storage for the dominant-axis projection that drives every predicate
{
    Box3f box;
    int pointsSize = 0;
    for ( const auto& loop : loops )
    {
        for ( EdgeId e : loop )
            box.include( mesh.orgPnt( e ) );
        if ( loop.size() >= 3 )
            pointsSize += int( loop.size() );
    }

    // drop the axis most aligned with the normal; order the two kept axes so the 2D ccw of the
    // projection equals the 3D orientation around +normal (swap them when normal points the other way)
    int dropAx = 0;
    for ( int i = 1; i < 3; ++i )
        if ( normal[i] * normal[i] > normal[dropAx] * normal[dropAx] )
            dropAx = i;
    int kx = ( dropAx + 1 ) % 3;
    int ky = ( dropAx + 2 ) % 3;
    if ( normal[dropAx] < 0 )
        std::swap( kx, ky );

    pts2->clear();
    pts2->reserve( pointsSize );
    auto toInt = getToIntConverter( Box3d( box ) ); // Vector3f -> Vector3i

    SweepLinePredicates p;
    setPts2Predicates( p, pts2 );
    p.addInputPoint = [&mesh, &loops, pts2, toInt, kx, ky] ( VertId v, int contourId, int pointId )
    {
        const Vector3i q = toInt( mesh.orgPnt( loops[contourId][pointId] ) );
        pts2->autoResizeSet( v, Vector2i( q[kx], q[ky] ) );
    };
    p.addIntersectionPoint = [pts2] ( VertId v, VertId a, VertId b, VertId c, VertId d )
    {
        pts2->autoResizeSet( v, findSegmentSegmentIntersectionPrecise( ( *pts2 )[a], ( *pts2 )[b], ( *pts2 )[c], ( *pts2 )[d] ) );
    };
    // disjoint triangulation creates no output intersection vertices, so every output vertex is an input
    // vertex: restore its exact original mesh position via the (already-built) holeVertIds identity map
    p.point = [&mesh, &holeVertIds] ( VertId v )
    {
        return mesh.points[holeSourceVertId( holeVertIds, v )];
    };
    return p;
}

int findClosestToFront( const MeshTopology& tp, const SweepLinePredicates& predicates,
    const std::vector<EdgeId>& edges, bool left )
{
    if ( edges.size() == 2 )
        return 1;
    const VertId org = tp.org( edges[1] );
    VertId baseId; // origin of the reference ray; invalid => the ray points against the sweep
    if ( edges[0] )
    {
        auto dest = tp.dest( edges[0] );
        for ( int i = 1; i < edges.size(); ++i )
        {
            if ( dest == tp.dest( edges[i] ) )
                return i;
        }
        baseId = dest;
    }
    // orientation of vertex x around org relative to the reference ray
    auto ccwFromBase = [&] ( VertId x )
    {
        return baseId.valid() ? predicates.ccw( baseId, x, org ) : predicates.ccwFromBehind( x, org );
    };
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
        const VertId destI = tp.dest( edges[i] );
        bool ccwBI = ccwFromBase( destI );
        int nextI = getNextI( i, ccwBI );

        const VertId destNextI = tp.dest( edges[nextI] );
        bool ccwBIn = ccwFromBase( destNextI );

        if ( ccwBI && !ccwBIn )
            return left ? i : nextI;
        if ( !ccwBI && ccwBIn )
            return left ? nextI : i;

        bool ccwIIn = predicates.ccw( destI, destNextI, org );

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


struct SweepLineParams
{
    /// if holesVertId is null - merge all vertices with same coordinates
    /// otherwise only merge the ones with same initial vertId
    const HolesVertIds* holesVertId{ nullptr };

    /// if not set - adds new vertices at intersection points
    /// otherwise aborts
    bool abortWhenIntersect{ false };

    WindingMode windingMode{ WindingMode::NonZero };

    /// if set do not do real triangulation, just marks inside faces as present
    bool needOutline{ false };

    /// one can disable merge for identical vertices, merge is useful on symbol contours
    bool allowMerge{ true };

    /// optional out EdgePaths that corresponds to initial contours
    std::vector<EdgePath>* outBoundaries{ nullptr };

    /// optional out per-face winding numbers
    Vector<int, FaceId>* outFaceWinding{ nullptr };
};

// the sweep-line triangulator
class SweepLineQueue
{
public:
    struct Cache; // all the buffers reused between runs, defined below

    // the queue works on the given cache, reusing the capacity of the buffers left there by previous runs
    explicit SweepLineQueue( Cache& cache ) : cache_( cache ) {}

    // resets the state of the previous run, if any (all cache buffers keep their capacity),
    // and makes initial mesh which simply contain input contours as edges
    // if params.holesVertId is null - merge all vertices with same coordinates
    // otherwise only merge the ones with same initial vertId
    void init( SweepLinePredicates predicates, std::vector<int> contourSizes, const SweepLineParams& params );

    size_t vertSize() const { return cache_.tp.vertSize(); }
    std::optional<Mesh> run( IntersectionsMap* interMap = nullptr );

    // same as run(), but does not create a Mesh: the result connectivity stays in the internal
    // cached topology (so its buffers are reused by the next run) and a pointer to it is returned,
    // valid until the next operation on this queue; nullptr if contours intersect while forbidden
    MeshTopology* runTopology( IntersectionsMap* interMap = nullptr );

    bool findIntersections();
    void injectIntersections( IntersectionsMap* interMap );
    void makeMonotone();
    void triangulate();
private:
    SweepLinePredicates predicates_;
    SweepLineParams params_;

// INITIALIZATION CLASS BLOCK
    // make base mesh only containing input contours as edge loops
    void initMeshByContours_( const std::vector<int>& contourSizes );
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
    void calculateWinding_();

    void triangulateMonotoneBlock_( EdgeId holeEdgeId );

// INTERSECTION CLASS BLOCK
    struct Intersection
    {
        EdgeId lower;
        EdgeId upper;
        VertId vId;
    };
    void setupStartVertices_();
    // index of next `startVerts`
    int startVertIndex_{ 0 };
    // index of next `sortedVerts`
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
    enum class EventType
    {
        Start, // item from `cache_.startVerts`
        Destination, // one of the `cache_.activeSweepEdges` destination vertices
        Intersection // intersection of two edges from `cache_.activeSweepEdges`
    };
    struct Event
    {
        // type of event
        EventType type{ EventType::Start };
        // EventType::Start - position to inject start edges
        // EventType::Destination - id of lowest edge (with this destenation) in `cache_.activeSweepEdges`
        // EventType::Intersection - id of lowest edge (with this intersection) in `cache_.activeSweepEdges`
        int index{ -1 }; // -1 means that we finished queue
        // return true if event is valid
        operator bool() const { return index != -1; }
    };
    // get next queue element
    Event getNext_();

    void invalidateIntersection_( int indexLower );
    bool isIntersectionValid_( int indexLower );

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
    // whether segments (aOrg,aDest) and (bOrg,bDest) properly intersect, via the injected ccw
    bool doSegmentSegmentIntersect_( VertId aOrg, VertId aDest, VertId bOrg, VertId bDest ) const;
    void checkIntersection_( int index, bool lower );
    void checkIntersection_( int indexLower );

public:
    // all the buffers the sweep-line triangulation reuses between runs sharing the cache;
    // the only implementation of ISweepLineCache
    struct Cache final : public ISweepLineCache
    {
        MeshTopology tp;
        VertCoords pointsCache; // scratch positions of tp's vertices for the Delone flips in triangulate()
        // storage for the projected points of the predicates built for the runs on this cache
        std::shared_ptr<Vector<Vector2i, VertId>> pts2Buffer = std::make_shared<Vector<Vector2i, VertId>>();
        Vector<EdgeWindingInfo, UndirectedEdgeId> windingInfo;
        std::vector<int> reflexChainCache;
        std::vector<Intersection> intersections;
        // scratch bitset of setupStartVertices_()
        VertBitSet startVerticesCache;
        // sorted vertices with no left-going edges
        std::vector<VertId> startVerts;
        std::vector<EdgeId> startVertLowestRight;
        // sorted vertices
        std::vector<VertId> sortedVerts;
        // edges that are intersected by sweep line ordered by position
        std::vector<SweepEdgeInfo> activeSweepEdges;
        // ordered events after intersection stage
        std::vector<Event> events;
        std::vector<SweepEdgeInfo> rightGoingCache;
        std::vector<EdgeId> findClosestCache;
        IntersectionMap intersectionsMap; // needed to prevent recreation of same vertices multiple times
    };

private:
    Cache& cache_;
};

ISweepLineCache::~ISweepLineCache() = default;

std::unique_ptr<ISweepLineCache> makeSweepLineCache()
{
    return std::make_unique<SweepLineQueue::Cache>();
}

void SweepLineQueue::init( SweepLinePredicates predicates, std::vector<int> contourSizes, const SweepLineParams& params )
{
    predicates_ = std::move( predicates );
    params_ = params;
    stage_ = Stage::Init;
    cache_.tp.clear(); // empty husk if the previous run() moved it out, filled if runTopology() kept it
    cache_.windingInfo.clear(); // stale winding modifiers would leak into this run through resize()
    cache_.intersections.clear();
    cache_.intersectionsMap.clear(); // keyed by the previous run's edge ids
    cache_.startVerts.clear();
    cache_.startVertLowestRight.clear();
    startVertIndex_ = 0;
    cache_.sortedVerts.clear(); // mergeSamePoints_() appends
    sortedVertIndex_ = 0;
    cache_.activeSweepEdges.clear();
    cache_.events.clear();

    initMeshByContours_( contourSizes );
    mergeSamePoints_( params.holesVertId );
    setupStartVertices_();
}

std::optional<MR::Mesh> SweepLineQueue::run( IntersectionsMap* interMap )
{
    if ( !runTopology( interMap ) )
        return {};
    // materialize the result, donating the cached buffers to it
    Mesh mesh;
    mesh.topology = std::move( cache_.tp );
    mesh.points = std::move( cache_.pointsCache );
    return mesh;
}

MeshTopology* SweepLineQueue::runTopology( IntersectionsMap* interMap )
{
    MR_TIMER;
    if ( !findIntersections() )
        return nullptr;
    injectIntersections( interMap );
    makeMonotone();
    triangulate();
    return &cache_.tp;
}

bool SweepLineQueue::findIntersections()
{
    MR_TIMER;
    stage_ = Stage::Intersections;
    cache_.events.clear();
    cache_.events.reserve( cache_.tp.numValidVerts() * 2 );
    while ( auto event = getNext_() )
    {
        if ( event.type == EventType::Start )
            processStartEvent_( event.index );
        else if ( event.type == EventType::Destination )
            processDestenationEvent_( event.index );
        else if ( event.type == EventType::Intersection )
        {
            if ( params_.abortWhenIntersect )
                return false;
            processIntersectionEvent_( event.index );
        }
        cache_.events.push_back( event );
    }
    return true;
}

void SweepLineQueue::injectIntersections( IntersectionsMap* interMap )
{
    MR_TIMER;

    if ( interMap )
        interMap->map.resize( cache_.intersections.size() );

    cache_.windingInfo.resize( cache_.windingInfo.size() + cache_.intersections.size() * 2 );
    Vector<EdgeId, UndirectedEdgeId> oldToFirstNewEdgeMap( cache_.tp.undirectedEdgeSize() );

    if ( interMap )
    {
        // create mapping if needed
        for ( const auto& inter : cache_.intersections )
        {
            auto ind = size_t( inter.vId ) - interMap->shift;
            assert( ind < interMap->map.size() );
            auto& mapVal = interMap->map[ind];
            mapVal.lOrg = cache_.tp.org( inter.lower );
            mapVal.lDest = cache_.tp.dest( inter.lower );
            mapVal.uOrg = cache_.tp.org( inter.upper );
            mapVal.uDest = cache_.tp.dest( inter.upper );

            auto iP = predicates_.point( inter.vId );
            auto lO = predicates_.point( mapVal.lOrg );
            auto lD = predicates_.point( mapVal.lDest );
            auto uO = predicates_.point( mapVal.uOrg );
            auto uD = predicates_.point( mapVal.uDest );
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

    for ( const auto& inter : cache_.intersections )
    {
        // split edges
        // set new edge ids to the left and save old to the right
        // because of intersections order

        // prev lower
        auto pl = cache_.tp.prev( inter.lower );
        // lower left
        auto ll = cache_.tp.makeEdge();
        if ( inter.lower.odd() )
            ll = ll.sym(); // oddity should stay the same (for winding number)
        cache_.tp.splice( pl, inter.lower );
        cache_.tp.splice( pl, ll );
        cache_.tp.splice( inter.lower, ll.sym() );

        // prev upper
        auto pu = cache_.tp.prev( inter.upper );
        // upper left
        auto ul = cache_.tp.makeEdge();
        if ( inter.upper.odd() )
            ul = ul.sym(); // oddity should stay the same (for winding number)

        cache_.tp.splice( pu, inter.upper );
        cache_.tp.splice( pu, ul );

        cache_.tp.splice( inter.lower, ul.sym() );
        cache_.tp.splice( ll.sym(), inter.upper );

        cache_.tp.setOrg( inter.upper, inter.vId );

        // winding modifiers of new parts should be same as old parts
        cache_.windingInfo[ll.undirected()].windingModifier = cache_.windingInfo[inter.lower.undirected()].windingModifier;
        cache_.windingInfo[ul.undirected()].windingModifier = cache_.windingInfo[inter.upper.undirected()].windingModifier;

        auto& otfnL = oldToFirstNewEdgeMap[inter.lower.undirected()];
        if ( !otfnL )
            otfnL = ll;
        auto& otfnU = oldToFirstNewEdgeMap[inter.upper.undirected()];
        if ( !otfnU )
            otfnU = ul;
    }
    for ( auto& e : cache_.startVertLowestRight )
        if ( auto newE = oldToFirstNewEdgeMap[e.undirected()] )
            e = newE;
}

void SweepLineQueue::makeMonotone()
{
    MR_TIMER;
    stage_ = Stage::Monotonation;
    startVertIndex_ = 0;
    sortedVertIndex_ = 0;
    for ( auto event : cache_.events )
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
    if ( !params_.needOutline )
        cache_.reflexChainCache.reserve( 256 ); // reserve once to have less allocations later
    for ( auto e : undirectedEdges( cache_.tp ) )
    {
        if ( e >= cache_.windingInfo.size() )
            continue;
        const auto& windInfo = cache_.windingInfo[e];
        if ( !windInfo.inside( params_.windingMode ) )
            continue;
        auto dirE = EdgeId( e << 1 );
        if ( !windInfo.rightGoing )
            dirE = dirE.sym();
        if ( cache_.tp.left( dirE ) )
            continue;

        const auto firstBlockFace = FaceId( cache_.tp.faceSize() );
        if ( !params_.needOutline )
            triangulateMonotoneBlock_( dirE ); // triangulate
        else
            cache_.tp.setLeft( dirE, cache_.tp.addFaceId() ); // mark present
        if ( params_.outFaceWinding ) // all faces of one monotone block are in the region with same winding number
            params_.outFaceWinding->autoResizeSet( firstBlockFace, cache_.tp.faceSize() - firstBlockFace, windInfo.winding );
    }
    cache_.pointsCache.resize( cache_.tp.vertSize() );
    BitSetParallelFor( cache_.tp.getValidVerts(), [&] ( VertId v )
    {
        cache_.pointsCache[v] = predicates_.point( v );
    } );
    // Delone flips could cross a contour edge between two inside regions and smear the face winding map
    if ( !params_.needOutline && !params_.outFaceWinding )
    {
        // borrow the cached topology and points into a temporary mesh for the flips (moves, no copies)
        Mesh mesh;
        mesh.topology = std::move( cache_.tp );
        mesh.points = std::move( cache_.pointsCache );
        makeDeloneEdgeFlips( mesh, {}, 300 );
        cache_.tp = std::move( mesh.topology );
        cache_.pointsCache = std::move( mesh.points );
    }
}

void SweepLineQueue::setupStartVertices_()
{
    auto& startVertices = cache_.startVerticesCache;
    startVertices.clear();
    startVertices.resize( cache_.tp.vertSize() );
    BitSetParallelFor( cache_.tp.getValidVerts(), [&] ( VertId v )
    {
        bool startVert = true;
        for ( auto e : orgRing( cache_.tp, v ) )
        {
            if ( predicates_.less( cache_.tp.dest( e ), v ) )
            {
                startVert = false;
                break;
            }
        }
        if ( startVert )
            startVertices.set( v );
    } );
    cache_.startVerts.resize( startVertices.count() );
    cache_.startVertLowestRight.resize( cache_.startVerts.size() );
    int i = 0;
    for ( auto v : startVertices )
        cache_.startVerts[i++] = v;

    std::sort( cache_.startVerts.begin(), cache_.startVerts.end(), [&] ( VertId l, VertId r )
    {
        return predicates_.less( l, r );
    } );
}

SweepLineQueue::Event SweepLineQueue::getNext_()
{
    Event outEvent;
    int minInterIndex = -1;

    VertId nextVertId;
    for ( ; sortedVertIndex_ < cache_.sortedVerts.size();)
    {
        nextVertId = cache_.sortedVerts[sortedVertIndex_];
        if ( cache_.tp.hasVert( nextVertId ) )
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
    for ( int i = 0; i < cache_.activeSweepEdges.size(); ++i )
    {
        const auto& activeSweep = cache_.activeSweepEdges[i];
        VertId destId = cache_.tp.dest( activeSweep.edgeId );
        if ( !minDestId && destId == nextVertId )
        {
            minDestId = destId; // we need first
            outEvent.type = EventType::Destination;
            outEvent.index = i;
        }
        if ( stage_ != Stage::Intersections || !activeSweep.upperInfo.interVertId )
            continue;
        if ( !minInter || predicates_.less( activeSweep.upperInfo.interVertId, minInter ) )
        {
            minInter = activeSweep.upperInfo.interVertId;
            minInterIndex = i;
        }
    }

    if ( minInter )
    {
        if ( cache_.tp.dest( cache_.activeSweepEdges[minInterIndex].edgeId ) == nextVertId ||
            cache_.tp.dest( cache_.activeSweepEdges[minInterIndex + 1].edgeId ) == nextVertId ||
            predicates_.less( minInter, nextVertId ) )
        {
            outEvent.type = EventType::Intersection;
            outEvent.index = minInterIndex;
            nextVertId = {};
        }
    }

    if ( startVertIndex_ < cache_.startVerts.size() )
    {
        if ( nextVertId == cache_.startVerts[startVertIndex_] )
        {
            outEvent.type = EventType::Start;
            outEvent.index = findStartIndex_();
        }
    }

    return outEvent;
}

void SweepLineQueue::invalidateIntersection_( int indexLower )
{
    if ( indexLower >= 0 && indexLower < cache_.activeSweepEdges.size() )
        cache_.activeSweepEdges[indexLower].upperInfo.interVertId = {};
    if ( indexLower + 1 >= 0 && indexLower + 1 < cache_.activeSweepEdges.size() )
        cache_.activeSweepEdges[indexLower + 1].lowerInfo.interVertId = {};
}

bool SweepLineQueue::isIntersectionValid_( int indexLower )
{
    if ( indexLower < 0 || indexLower + 1 >= cache_.activeSweepEdges.size() )
        return false;
    if ( !cache_.activeSweepEdges[indexLower].upperInfo.interVertId )
        return false;
    return cache_.activeSweepEdges[indexLower].upperInfo.interVertId == cache_.activeSweepEdges[indexLower + 1].lowerInfo.interVertId;
}

int SweepLineQueue::findStartIndex_()
{
    int activeVPosition{ INT_MAX };// index of first edge, under activeV (INT_MAX - all edges are lower, -1 - all edges are upper)
    const VertId activeV = cache_.startVerts[startVertIndex_];
    for ( int i = 0; i < cache_.activeSweepEdges.size(); ++i )
    {
        const VertId org = cache_.tp.org( cache_.activeSweepEdges[i].edgeId );
        const VertId dest = cache_.tp.dest( cache_.activeSweepEdges[i].edgeId );

        if ( activeVPosition == INT_MAX && predicates_.ccw( org, activeV, dest ) )
            activeVPosition = i - 1;
    }

    return activeVPosition == INT_MAX ? int( cache_.activeSweepEdges.size() ) : activeVPosition + 1;
}

void SweepLineQueue::updateStartRightGoingCache_()
{
    cache_.rightGoingCache.clear();
    if ( stage_ == Stage::Intersections )
    {
        cache_.findClosestCache.clear();
        cache_.findClosestCache.emplace_back( EdgeId{} );
    }
    for ( auto e : orgRing( cache_.tp, cache_.startVerts[startVertIndex_] ) )
    {
        cache_.rightGoingCache.emplace_back( SweepEdgeInfo{ .edgeId = e } );
        if ( stage_ == Stage::Intersections )
            cache_.findClosestCache.push_back( e );
    }

    int pos = -1;
    if ( stage_ == Stage::Intersections )
    {
        pos = findClosestToFront( cache_.tp, predicates_, cache_.findClosestCache, true ) - 1;
        assert( pos > -1 );
        cache_.startVertLowestRight[startVertIndex_] = cache_.rightGoingCache[pos].edgeId;
    }
    else
    {
        for ( int i = 0; i < cache_.rightGoingCache.size(); ++i )
        {
            if ( cache_.rightGoingCache[i].edgeId != cache_.startVertLowestRight[startVertIndex_] )
                continue;
            pos = i;
            break;
        }
        assert( pos > -1 );
    }

    std::rotate( cache_.rightGoingCache.begin(), cache_.rightGoingCache.begin() + pos, cache_.rightGoingCache.end() );
}

void SweepLineQueue::processStartEvent_( int index )
{
    updateStartRightGoingCache_();

    if ( stage_ == Stage::Intersections )
    {
        invalidateIntersection_( index - 1 );
    }

    if ( stage_ == Stage::Monotonation && index > 0 && index < cache_.activeSweepEdges.size() &&
        cache_.windingInfo[cache_.activeSweepEdges[index - 1].edgeId.undirected()].inside( params_.windingMode ) )
    {
        // find helper:
        // id of rightmost left vertex (it's lower edge) closest to active vertex
        // close to `helper` described here : https://www.cs.umd.edu/class/spring2020/cmsc754/Lects/lect05-triangulate.pdf
        EdgeId helperId;
        auto& lowerLone = cache_.activeSweepEdges[index - 1].upperInfo.loneEdgeId;
        auto& upperLone = cache_.activeSweepEdges[index].lowerInfo.loneEdgeId;
        assert( lowerLone == upperLone );
        if ( lowerLone )
        {
            helperId = lowerLone;
            lowerLone = upperLone = {};
        }
        else
        {
            auto lowerOrg = cache_.tp.org( cache_.activeSweepEdges[index - 1].edgeId );
            auto upperOrg = cache_.tp.org( cache_.activeSweepEdges[index].edgeId );
            if ( predicates_.less( lowerOrg, upperOrg ) )
                helperId = cache_.tp.prev( cache_.activeSweepEdges[index].edgeId );
            else
                helperId = cache_.activeSweepEdges[index - 1].edgeId;
        }
        assert( helperId );

        auto newEdge = cache_.tp.makeEdge();
        if ( cache_.activeSweepEdges[index - 1].edgeId.odd() )
            newEdge = newEdge.sym();
        cache_.tp.splice( helperId, newEdge );
        cache_.tp.splice( cache_.rightGoingCache.back().edgeId, newEdge.sym() );

        cache_.windingInfo.autoResizeSet( newEdge.undirected(), cache_.windingInfo[cache_.activeSweepEdges[index - 1].edgeId.undirected()] );
    }

    cache_.activeSweepEdges.insert( cache_.activeSweepEdges.begin() + index, cache_.rightGoingCache.begin(), cache_.rightGoingCache.end() );

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
    for ( int i = minIndex + 1; i < cache_.activeSweepEdges.size(); ++i )
    {
        if ( cache_.tp.dest( cache_.activeSweepEdges[index].edgeId ) != cache_.tp.dest( cache_.activeSweepEdges[i].edgeId ) )
            break;
        maxIndex = i;
    }
    cache_.rightGoingCache.clear();
    for ( auto e : orgRing0( cache_.tp, cache_.activeSweepEdges[minIndex].edgeId.sym() ) )
    {
        if ( e == cache_.activeSweepEdges[maxIndex].edgeId.sym() )
            break;
        cache_.rightGoingCache.emplace_back( SweepEdgeInfo{ .edgeId = e } );
    }
    int numLeft = maxIndex - minIndex + 1;
    int numRight = int( cache_.rightGoingCache.size() );
    EdgeId lowestLeft = cache_.activeSweepEdges[minIndex].edgeId;
    if ( stage_ == Stage::Monotonation )
    {
        // connect with prev lone if needed
        for ( int i = std::max( 0, minIndex - 1 ); i < std::min( maxIndex + 1, int( cache_.activeSweepEdges.size() ) - 1 ); ++i )
        {
            auto& lowerLone = cache_.activeSweepEdges[i].upperInfo.loneEdgeId;
            auto& upperLone = cache_.activeSweepEdges[i + 1].lowerInfo.loneEdgeId;
            assert( lowerLone == upperLone );
            if ( !lowerLone )
                continue;

            EdgeId connectorEdgeId;
            if ( i < maxIndex )
                connectorEdgeId = cache_.activeSweepEdges[i + 1].edgeId.sym();
            else
                connectorEdgeId = cache_.tp.prev( cache_.activeSweepEdges[i].edgeId.sym() );

            auto newEdge = cache_.tp.makeEdge();
            if ( cache_.activeSweepEdges[i].edgeId.odd() )
                newEdge = newEdge.sym();
            cache_.tp.splice( lowerLone, newEdge );
            cache_.tp.splice( connectorEdgeId, newEdge.sym() );

            lowerLone = upperLone = {};

            cache_.windingInfo.autoResizeSet( newEdge.undirected(), cache_.windingInfo[cache_.activeSweepEdges[i].edgeId.undirected()] );
            if ( i == minIndex - 1 )
                lowestLeft = newEdge;
        }
    }
    if ( numRight == 0 )
    {
        if ( stage_ == Stage::Monotonation && minIndex > 0 && maxIndex + 1 < cache_.activeSweepEdges.size() &&
            cache_.windingInfo[cache_.activeSweepEdges[minIndex - 1].edgeId.undirected()].inside( params_.windingMode ) )
        {
            cache_.activeSweepEdges[minIndex - 1].upperInfo.loneEdgeId = lowestLeft.sym();
            cache_.activeSweepEdges[maxIndex + 1].lowerInfo.loneEdgeId = lowestLeft.sym();
        }
        cache_.activeSweepEdges.erase( cache_.activeSweepEdges.begin() + minIndex, cache_.activeSweepEdges.begin() + maxIndex + 1 );
        if ( stage_ == Stage::Intersections )
        {
            checkIntersection_( minIndex - 1, false );
        }
    }
    else
    {
        for ( int i = minIndex; i < minIndex + std::min( numLeft, numRight ); ++i )
        {
            assert( i < cache_.activeSweepEdges.size() );
            cache_.activeSweepEdges[i] = cache_.rightGoingCache[i - minIndex];
        }
        if ( numLeft > numRight )
            cache_.activeSweepEdges.erase( cache_.activeSweepEdges.begin() + minIndex + numRight, cache_.activeSweepEdges.begin() + maxIndex + 1 );
        else if ( numLeft < numRight )
            cache_.activeSweepEdges.insert( cache_.activeSweepEdges.begin() + maxIndex + 1, cache_.rightGoingCache.begin() + numLeft, cache_.rightGoingCache.end() );

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
        cache_.intersections.emplace_back( Intersection{
            .lower = cache_.activeSweepEdges[index].edgeId,
            .upper = cache_.activeSweepEdges[index + 1].edgeId } );
    }
    invalidateIntersection_( index );
    if ( !isValid )
        return;

    auto minEdgeId = std::min( cache_.activeSweepEdges[index].edgeId, cache_.activeSweepEdges[index + 1].edgeId );
    auto maxEdgeId = std::max( cache_.activeSweepEdges[index].edgeId, cache_.activeSweepEdges[index + 1].edgeId );

    auto& interInfo = cache_.intersectionsMap.at( { minEdgeId,maxEdgeId } );
    assert( !interInfo.processed );
    interInfo.processed = true;
    cache_.intersections.back().vId = interInfo.vId;

    invalidateIntersection_( index - 1 );
    invalidateIntersection_( index + 1 );

    std::swap( cache_.activeSweepEdges[index], cache_.activeSweepEdges[index + 1] );

    checkIntersection_( index, true );
    checkIntersection_( index + 1, false );
}

void SweepLineQueue::checkIntersection_( int index, bool lower )
{
    if ( index < 0 || index >= cache_.activeSweepEdges.size() )
        return;
    if ( lower && index == 0 )
        return;
    if ( !lower && index + 1 == cache_.activeSweepEdges.size() )
        return;
    if ( lower && index >= 1 )
        return checkIntersection_( index - 1 );
    if ( !lower && index + 1 < cache_.activeSweepEdges.size() )
        return checkIntersection_( index );
}

bool SweepLineQueue::doSegmentSegmentIntersect_( VertId aOrg, VertId aDest, VertId bOrg, VertId bDest ) const
{
    // segments intersect iff each separates the endpoints of the other;
    // predicate-based equivalent of MR::doSegmentSegmentIntersect (cIsLeftFromAB is unused here)
    const bool abc = predicates_.ccw( aOrg, aDest, bOrg );
    const bool abd = predicates_.ccw( aOrg, aDest, bDest );
    if ( abc == abd )
        return false; // segment (bOrg,bDest) is on one side of line (aOrg,aDest)
    const bool cda = predicates_.ccw( bOrg, bDest, aOrg );
    const bool cdb = predicates_.ccw( bOrg, bDest, aDest );
    return cda != cdb; // equal => segment (aOrg,aDest) is on one side of line (bOrg,bDest)
}

void SweepLineQueue::checkIntersection_( int i )
{
    assert( i >= 0 && i + 1 < cache_.activeSweepEdges.size() );

    const VertId org1 = cache_.tp.org( cache_.activeSweepEdges[i].edgeId );
    const VertId dest1 = cache_.tp.dest( cache_.activeSweepEdges[i].edgeId );
    const VertId org2 = cache_.tp.org( cache_.activeSweepEdges[i + 1].edgeId );
    const VertId dest2 = cache_.tp.dest( cache_.activeSweepEdges[i + 1].edgeId );
    bool canIntersect = org1 != org2 && dest1 != dest2;
    if ( !canIntersect || !org1 || !org2 || !dest1 || !dest2 )
        return;

    if ( !doSegmentSegmentIntersect_( org1, dest1, org2, dest2 ) )
        return;

    auto minEdgeId = std::min( cache_.activeSweepEdges[i].edgeId, cache_.activeSweepEdges[i + 1].edgeId );
    auto maxEdgeId = std::max( cache_.activeSweepEdges[i].edgeId, cache_.activeSweepEdges[i + 1].edgeId );
    auto& interInfo = cache_.intersectionsMap[{minEdgeId, maxEdgeId}];
    if ( !interInfo )
    {
        interInfo.vId = cache_.tp.addVertId();
        predicates_.addIntersectionPoint( interInfo.vId, org1, dest1, org2, dest2 );
    }
    else if ( interInfo.processed )
        return;

    cache_.activeSweepEdges[i].upperInfo.interVertId = interInfo.vId;
    cache_.activeSweepEdges[i + 1].lowerInfo.interVertId = interInfo.vId;
}

void SweepLineQueue::initMeshByContours_( const std::vector<int>& contourSizes )
{
    MR_TIMER;
    for ( int contourId = 0; contourId < int( contourSizes.size() ); ++contourId )
    {
        if ( contourSizes[contourId] > 3 )
        {
            for ( int pointId = 0; pointId + 1 < contourSizes[contourId]; ++pointId )
            {
                VertId v = cache_.tp.addVertId();
                predicates_.addInputPoint( v, contourId, pointId );
            }
        }
    }

    int boundId = -1;
    if ( params_.outBoundaries )
        params_.outBoundaries->resize( contourSizes.size() );

    int firstVert = 0;
    for ( int contSize : contourSizes )
    {
        ++boundId;
        if ( contSize <= 3 )
            continue;

        int size = contSize - 1;

        if ( params_.outBoundaries )
            ( *params_.outBoundaries )[boundId].resize( size );

        for ( int i = 0; i < size; ++i )
        {
            auto newEdgeId = cache_.tp.makeEdge();
            cache_.tp.setOrg( newEdgeId, VertId( firstVert + i ) );
            if ( params_.outBoundaries )
                ( *params_.outBoundaries )[boundId][i] = newEdgeId;
        }
        const auto& edgePerVert = cache_.tp.edgePerVertex();
        for ( int i = 0; i < size; ++i )
            cache_.tp.splice( edgePerVert[VertId( firstVert + i )], edgePerVert[VertId( firstVert + ( ( i + int( size ) - 1 ) % size ) )].sym() );
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
    cache_.sortedVerts.reserve( cache_.tp.vertSize() );
    for ( int i = 0; i < cache_.tp.vertSize(); ++i )
        cache_.sortedVerts.emplace_back( VertId( i ) );
    if ( !holesVertId )
        tbb::parallel_sort( cache_.sortedVerts.begin(), cache_.sortedVerts.end(), [&] ( VertId l, VertId r ) { return predicates_.less( l, r ); } );
    else
    {
        tbb::parallel_sort( cache_.sortedVerts.begin(), cache_.sortedVerts.end(), [&] ( VertId l, VertId r )
        {
            if ( predicates_.less2d( l, r ) )
                return true;
            if ( predicates_.less2d( r, l ) )
                return false;
            return findRealVertId( l ) < findRealVertId( r );
        } );
    }

    if ( !params_.allowMerge )
    {
        cache_.windingInfo.resize( cache_.tp.undirectedEdgeSize() );
        return;
    }

    int prevUnique = 0;
    for ( int i = 1; i < cache_.sortedVerts.size(); ++i )
    {
        bool sameIntCoord = predicates_.samePos( cache_.sortedVerts[i], cache_.sortedVerts[prevUnique] );
        if ( !sameIntCoord )
        {
            prevUnique = i;
            continue;
        }
        // if same coords
        if ( !holesVertId || findRealVertId( cache_.sortedVerts[prevUnique] ) == findRealVertId( cache_.sortedVerts[i] ) )
            mergeSinglePare_( cache_.sortedVerts[prevUnique], cache_.sortedVerts[i] );
    }

    if ( holesVertId ) // sort with correct indices in case of other way sort before
        tbb::parallel_sort( cache_.sortedVerts.begin(), cache_.sortedVerts.end(), [&] ( VertId l, VertId r ) { return predicates_.less( l, r ); } );

    cache_.windingInfo.resize( cache_.tp.undirectedEdgeSize() );
    if ( !params_.abortWhenIntersect || !holesVertId )
        removeMultipleAfterMerge_();
}

void SweepLineQueue::mergeSinglePare_( VertId unique, VertId same )
{
    std::vector<EdgeId> sameEdges;
    int sameToUniqueEdgeIndex{ -1 };
    int i = 0;
    for ( auto eSame : orgRing( cache_.tp, same ) )
    {
        sameEdges.push_back( eSame );
        if ( cache_.tp.dest( eSame ) == unique )
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
        cache_.tp.splice( cache_.tp.prev( e ), e );
        cache_.tp.splice( cache_.tp.prev( e.sym() ), e.sym() );
        sameEdges.erase( sameEdges.begin() + sameToUniqueEdgeIndex );
        if ( sameEdges.empty() )
        {
            cache_.tp.setOrg( e, VertId{} ); // the "same" becomes invalid after removing of its only edge
            cache_.tp.setOrg( e.sym(), VertId{}); // the "same" becomes invalid after removing of its only edge
        }
    }

    for ( auto eSame : sameEdges )
    {
        cache_.findClosestCache.clear();
        cache_.findClosestCache.push_back( eSame );
        for ( auto eUnique : orgRing( cache_.tp, unique ) )
        {
            cache_.findClosestCache.emplace_back( eUnique );
        }
        auto minEUnique = cache_.findClosestCache[findClosestToFront( cache_.tp, predicates_, cache_.findClosestCache, false )];
        auto prev = cache_.tp.prev( eSame );
        if ( prev != eSame )
            cache_.tp.splice( prev, eSame );
        else
            cache_.tp.setOrg( eSame, VertId{} );
        cache_.tp.splice( minEUnique, eSame );
        if ( cache_.tp.dest( minEUnique ) == cache_.tp.dest( eSame ) )
        {
            auto& edgeInfo = cache_.windingInfo.autoResizeAt( minEUnique.undirected() );
            if ( edgeInfo.windingModifier == INT_MAX )
                edgeInfo.windingModifier = 1;
            bool uniqueIsOdd = minEUnique.odd();
            bool sameIsOdd = eSame.odd();
            edgeInfo.windingModifier += ( ( uniqueIsOdd == sameIsOdd ) ? 1 : -1 );
            cache_.tp.splice( cache_.tp.prev( eSame ), eSame );
            cache_.tp.splice( cache_.tp.prev( eSame.sym() ), eSame.sym() );
            if ( cache_.tp.next( minEUnique ) == minEUnique )
            {
                // invalidate lone edge
                cache_.tp.splice( cache_.tp.prev( minEUnique.sym() ), minEUnique.sym() );
                cache_.tp.setOrg( minEUnique, VertId{} );
                cache_.tp.setOrg( minEUnique.sym(), VertId{} );
            }
        }
    }
}

void SweepLineQueue::removeMultipleAfterMerge_()
{
    MR_TIMER;
    auto multiples = findMultipleEdges( cache_.tp ).value();
    for ( const auto& multiple : multiples )
    {
        std::vector<EdgeId> multiplesFromThis;
        for ( auto e : orgRing( cache_.tp, multiple.first ) )
        {
            if ( cache_.tp.dest( e ) == multiple.second )
                multiplesFromThis.push_back( e );
        }
        assert( multiplesFromThis.size() > 1 );

        if ( params_.outBoundaries )
        {
            auto& bounds = *params_.outBoundaries;
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

        auto& edgeInfo = cache_.windingInfo[multiplesFromThis.front().undirected()];
        edgeInfo.windingModifier = 1;
        bool uniqueIsOdd = int( multiplesFromThis.front() ) & 1;
        for ( int i = 1; i < multiplesFromThis.size(); ++i )
        {
            auto e = multiplesFromThis[i];
            bool isMEOdd = int( e ) & 1;
            edgeInfo.windingModifier += ( ( uniqueIsOdd == isMEOdd ) ? 1 : -1 );
            cache_.tp.splice( cache_.tp.prev( e ), e );
            cache_.tp.splice( cache_.tp.prev( e.sym() ), e.sym() );
            assert( cache_.tp.isLoneEdge( e ) );
        }
    }
}

void SweepLineQueue::calculateWinding_()
{
    int windingLast = 0;
    // recalculate winding number for active edges
    for ( const auto& e : cache_.activeSweepEdges )
    {
        auto& info = cache_.windingInfo[e.edgeId.undirected()];
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
    auto holeLoop = trackRightBoundaryLoop( cache_.tp, holeEdgeId );
    auto lessPred = [&] ( EdgeId l, EdgeId r )
    {
        return predicates_.less( cache_.tp.org( l ) , cache_.tp.org( r ) );
    };
    auto minMaxIt = std::minmax_element( holeLoop.begin(), holeLoop.end(), lessPred );

    int loopSize = int( holeLoop.size() );
    int minIndex = int( std::distance( holeLoop.begin(), minMaxIt.first ) );
    int maxIndex = int( std::distance( holeLoop.begin(), minMaxIt.second ) );
    auto nextLowerLoopInd = [&] ( int curIdx ) { return ( curIdx + 1 ) % loopSize; };
    auto nextUpperLoopInd = [&] ( int curIdx ) { return ( curIdx - 1 + loopSize ) % loopSize; };

    auto isReflex = [&] ( int prev, int cur, int next, bool lowerChain )
    {
        return predicates_.ccw( cache_.tp.org( holeLoop[prev] ), cache_.tp.org( holeLoop[next] ), cache_.tp.org( holeLoop[cur] ) ) == lowerChain;
    };

    auto addDiagonal = [&] ( int cur, int prev, bool lowerChain )->bool
    {
        auto& tp = cache_.tp;
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

    auto& reflexChain = cache_.reflexChainCache;
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

Mesh getOutlineMesh( const Contours2f& conts, IntersectionsMap* interMap /*= nullptr */, const BaseOutlineParameters& params )
{
    SweepLineQueue::Cache cache;
    SweepLineQueue triangulator( cache );
    triangulator.init( precisePredicates( conts, cache.pts2Buffer ), getContourSizes( conts ), { nullptr, false, params.innerType, true, params.allowMerge } );

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

Mesh getOutlineMesh( const Contours2d& contours, IntersectionsMap* interMap /*= nullptr */, const BaseOutlineParameters& params )
{
    const auto contsf = convertContours<Contours2f>( contours );
    return getOutlineMesh( contsf, interMap, params );
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
    const auto contsd = convertContours<Contours2d>( contours );
    return getOutline( contsd, params );
}

Mesh triangulateContours( const Contours2f& contours, const TriangulationParameters& params /*= {}*/ )
{
    if ( contours.empty() )
        return {};
    SweepLineQueue::Cache cache;
    SweepLineQueue triangulator( cache );
    triangulator.init( precisePredicates( contours, cache.pts2Buffer ), getContourSizes( contours ),
        { params.holeVertsIds, false, WindingMode::NonZero, false, true, nullptr, params.outFaceWinding } );
    if ( params.outInterMap )
        params.outInterMap->shift = triangulator.vertSize();
    auto res = triangulator.run( params.outInterMap );
    assert( res );
    if ( res )
        return std::move( *res );
    else
        return Mesh();
}

Mesh triangulateContours( const Contours2d& contours, const TriangulationParameters& params /*= {}*/ )
{
    const auto contsf = convertContours<Contours2f>( contours );
    return triangulateContours( contsf, params );
}

Mesh triangulateContours( const Contours2d& contours, const HolesVertIds* holeVertsIds )
{
    return triangulateContours( contours, { .holeVertsIds = holeVertsIds } );
}

Mesh triangulateContours( const Contours2f& contours, const HolesVertIds* holeVertsIds )
{
    return triangulateContours( contours, { .holeVertsIds = holeVertsIds } );
}

std::optional<Mesh> triangulateDisjointContours( const Contours2d& contours, const HolesVertIds* holeVertsIds /*= nullptr*/, std::vector<EdgePath>* outBoundaries /*= nullptr*/, ISweepLineCache* cache /*= nullptr*/ )
{
    const auto contsf = convertContours<Contours2f>( contours );
    return triangulateDisjointContours( contsf, holeVertsIds, outBoundaries, cache );
}

std::optional<Mesh> triangulateDisjointContours( const Contours2f& contours, const HolesVertIds* holeVertsIds /*= nullptr*/, std::vector<EdgePath>* outBoundaries /*= nullptr*/, ISweepLineCache* cache /*= nullptr*/ )
{
    if ( contours.empty() )
        return Mesh();
    std::optional<SweepLineQueue::Cache> localCache;
    auto& cacheImpl = cache ? static_cast<SweepLineQueue::Cache&>( *cache ) : localCache.emplace();
    SweepLineQueue triangulator( cacheImpl );
    triangulator.init( precisePredicates( contours, cacheImpl.pts2Buffer ), getContourSizes( contours ), { holeVertsIds, true, WindingMode::NonZero, false, true, outBoundaries } );
    return triangulator.run();
}

MeshTopology* triangulateDisjointContoursTopology( const Contours2f& contours, const HolesVertIds* holeVertsIds, std::vector<EdgePath>* outBoundaries, ISweepLineCache& cache )
{
    auto& cacheImpl = static_cast<SweepLineQueue::Cache&>( cache );
    SweepLineQueue triangulator( cacheImpl );
    triangulator.init( precisePredicates( contours, cacheImpl.pts2Buffer ), getContourSizes( contours ), { holeVertsIds, true, WindingMode::NonZero, false, true, outBoundaries } );
    return triangulator.runTopology();
}

// boundary loops may share mesh vertices: the identity map drives both vertex merging and exact
// point restoration (so the patch reuses the original mesh coordinates instead of projected ones)
static void initMeshSpace_( SweepLineQueue& triangulator, SweepLineQueue::Cache& cacheImpl, const Mesh& mesh,
    const EdgeLoops& loops, const Vector3f& normal, const HolesVertIds& holeVertIds, std::vector<EdgePath>* outBoundaries )
{
    std::vector<int> sizes( loops.size() );
    for ( int i = 0; i < int( loops.size() ); ++i )
        sizes[i] = int( loops[i].size() ) + 1; // +1 matches the queue's closed-contour convention (it allocates size-1 verts)
    triangulator.init( meshSpacePredicates( mesh, loops, normal, holeVertIds, cacheImpl.pts2Buffer ), std::move( sizes ),
        { &holeVertIds, true, WindingMode::NonZero, false, true, outBoundaries } );
}

std::optional<Mesh> triangulateDisjointContours( const Mesh& mesh, const EdgeLoops& loops, const Vector3f& normal, std::vector<EdgePath>* outBoundaries /*= nullptr*/ )
{
    if ( loops.empty() )
        return Mesh();
    const HolesVertIds holeVertIds = findHoleVertIdsByHoleEdges( mesh.topology, loops );
    SweepLineQueue::Cache cache;
    SweepLineQueue triangulator( cache );
    initMeshSpace_( triangulator, cache, mesh, loops, normal, holeVertIds, outBoundaries );
    return triangulator.run();
}

MeshTopology* triangulateDisjointContoursTopology( const Mesh& mesh, const EdgeLoops& loops, const Vector3f& normal, std::vector<EdgePath>* outBoundaries, ISweepLineCache& cache )
{
    if ( loops.empty() )
        return nullptr;
    const HolesVertIds holeVertIds = findHoleVertIdsByHoleEdges( mesh.topology, loops );
    auto& cacheImpl = static_cast<SweepLineQueue::Cache&>( cache );
    SweepLineQueue triangulator( cacheImpl );
    initMeshSpace_( triangulator, cacheImpl, mesh, loops, normal, holeVertIds, outBoundaries );
    return triangulator.runTopology();
}

}

}
