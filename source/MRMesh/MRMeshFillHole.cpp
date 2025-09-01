#include "MRMeshFillHole.h"
#include "MRMesh.h"
#include "MRBitSet.h"
#include "MRVector3.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRPlane3.h"
#include "MRMeshBuilder.h"
#include "MRMeshDelone.h"
#include "MRMarkedContour.h"
#include "MRGTest.h"
#include "MRParallelFor.h"
#include "MRPch/MRSpdlog.h"
#include <parallel_hashmap/phmap.h>
#include <queue>
#include <functional>

namespace MR
{

// This is structure for representing edge between two vertices on hole
// a,b - indices of edges
// prevA - index of best found triangle (it continue to edges [a,prevA] and [prevA,a]) (in case of fillHole)
// prevA, prevB - index of best found prev edge (prevA - index of vert on 'a' loop, prevB index of vert on 'b' loop) (in case of stithHoles)
// weight is sum of prev weight and metric value of this new edge
struct WeightedConn
{
    WeightedConn() = default;
    WeightedConn( int _a, int _b, double _weight, int _prevB = -1 ) :
        a{ _a }, b{ _b }, weight{ _weight }, prevB{ _prevB }{}

    int a{-1};
    int b{-1};
    double weight{DBL_MAX};

    int prevA{ -1 };
    int prevB{ -1 };

    bool hasPrev()const
    {
        return prevA != -1 && prevB != -1;
    }
};

bool operator<( const WeightedConn& left, const WeightedConn& right )
{
    return left.weight > right.weight;
}

typedef std::vector<std::vector<WeightedConn>> NewEdgesMap;

bool sameEdgeExists( const MeshTopology& topology, EdgeId e1Org, EdgeId e2Org )
{
    VertId org2 = topology.org( e2Org );
    for ( auto e : orgRing( topology, e1Org ) )
        if ( topology.dest( e ) == org2 )
            return true;
    return false;
}

bool holeHasDuplicateVerts( const MeshTopology& topology, EdgeId a )
{
    VertId max;
    for ( auto e : leftRing( topology, a ) )
    {
        auto v = topology.org( e );
        if ( v > max )
            max = v;
    }
    VertBitSet bitSet( max + 1 );
    for ( auto e : leftRing( topology, a ) )
    {
        auto v = topology.org( e );
        if ( bitSet.test( v ) )
            return true;
        bitSet.set( v );
    }
    return false;
}

void getOptimalSteps( std::vector<unsigned>& optimalSteps, unsigned start, unsigned steps, unsigned loopSize, int maxPolygonSubdivisions )
{
    optimalSteps.resize(0);
    --steps;
    if ( (int)steps <= maxPolygonSubdivisions )
    {
        for ( unsigned i = 0; i < steps; ++i )
            optimalSteps.push_back( ( start + i ) % loopSize );
        return;
    }

    for ( int i = 0; i < ( maxPolygonSubdivisions / 4 ); ++i )
        optimalSteps.push_back( ( start + i ) % loopSize );


    auto bigStep = ( steps - ( maxPolygonSubdivisions / 2 ) ) / ( maxPolygonSubdivisions / 2 );
    auto numBigSteps = ( maxPolygonSubdivisions / 2 );
    if ( bigStep < 2 )
    {
        bigStep = 2;
        numBigSteps = ( maxPolygonSubdivisions / 4 );
    }
    auto bigStepHalf = bigStep / 2;
    const auto newStart = start + bigStepHalf + ( maxPolygonSubdivisions / 4 ) - 1;
    for ( int i = 0; i < numBigSteps; ++i )
        optimalSteps.push_back( ( newStart + i * bigStep ) % loopSize );

    for ( int i = ( maxPolygonSubdivisions / 4 ) -1; i >=0 ; --i )
        optimalSteps.push_back( ( start + steps - i - 1 ) % loopSize );
}

// finds best candidate among all given steps
void getTriangulationWeights( const MeshTopology& topology, const NewEdgesMap& map, const EdgePath& loop,
    const FillHoleMetric& metrics, bool smoothBd,
    const std::vector<unsigned>& optimalStepsCache, WeightedConn& processedConn )
{
    for ( unsigned s = 0; s < optimalStepsCache.size(); ++s )
    {
        auto v = optimalStepsCache[s];
        const auto& abConn = map[processedConn.a][v];
        const auto& bcConn = map[v][processedConn.b];
        double weight = metrics.combineMetric( abConn.weight, bcConn.weight );
        if ( weight > processedConn.weight )
            continue;

        VertId aVert = topology.org( loop[processedConn.a] );
        VertId bVert = topology.org( loop[processedConn.b] );

        if ( aVert == bVert )
            continue;

        if ( metrics.triangleMetric )
        {
            auto triMetric = metrics.triangleMetric( aVert, topology.org( loop[v] ), bVert );
            weight = metrics.combineMetric( weight, triMetric );
        }
        if ( metrics.edgeMetric )
        {
            VertId leftVert;
            if ( abConn.hasPrev() )
                leftVert = topology.org( loop[abConn.prevA] );
            else if ( smoothBd && topology.right( loop[processedConn.a] ) )
                leftVert = topology.dest( topology.prev( loop[processedConn.a] ) );

            if ( leftVert )
            {
                auto edgeACMetric = metrics.edgeMetric( aVert, topology.org( loop[v] ), leftVert, bVert );
                weight = metrics.combineMetric( weight, edgeACMetric );
            }

            VertId rightVert;
            if ( bcConn.hasPrev() )
                rightVert = topology.org( loop[bcConn.prevA] );
            else if ( smoothBd && topology.right( loop[v] ) )
                rightVert = topology.dest( topology.prev( loop[v] ) );

            if ( rightVert )
            {
                auto edgeCBMetric = metrics.edgeMetric( topology.org( loop[v] ), bVert, rightVert, aVert );
                weight = metrics.combineMetric( weight, edgeCBMetric );
            }
        }

        if ( weight < processedConn.weight )
        {
            processedConn.weight = weight;
            processedConn.prevA = v; // In this case prevA describes chosen triangulation
        }
    }
}

struct MapPatchElement
{
    int a{ -1 };
    int b{ -1 };
    int newPrevA{ -1 };
};
using MapPatch = std::vector<MapPatchElement>;

// this function go backward by given triangulation and tries to fix multiple edges
// return false if triangulation has multiple edges that cannot be fixed
bool removeMultipleEdgesFromTriangulation( const MeshTopology& topology, const NewEdgesMap& map, const EdgePath& loop, const FillHoleMetric& metricRef, bool smoothBd,
    WeightedConn start, int maxPolygonSubdivisions, MapPatch& mapPatch )
{
    MR_TIMER;
    mapPatch.clear();
    HashSet<VertPair> edgesInTriangulation;
    auto testExistance = [&] ( int a, int b )->bool
    {
        auto dist = ( a - b + loop.size() ) % loop.size();
        if ( dist == 1 || dist + 1 == loop.size() )
            return false;
        auto aV = topology.org( loop[a] );
        auto bV = topology.org( loop[b] );
        if ( topology.findEdge( aV, bV ) )
            return true;
        if ( aV > bV )
            std::swap( aV, bV );
        auto it = edgesInTriangulation.find( { aV,bV } );
        return it != edgesInTriangulation.end();
    };
    std::vector<unsigned> optimalStepsCache( maxPolygonSubdivisions );
    std::queue<WeightedConn> newEdgesQueue;

    auto pushInQueue = [&] ( int a, int b )
    {
        VertId aV, bV;
        aV = topology.org( loop[a] );
        bV = topology.org( loop[b] );
        if ( aV > bV )
            std::swap( aV, bV );
        edgesInTriangulation.insert( { aV,bV } );
        newEdgesQueue.push( map[a][b] );
    };

    pushInQueue( start.a, start.b );
    while ( !newEdgesQueue.empty() )
    {
        start = std::move( newEdgesQueue.front() );
        newEdgesQueue.pop();
        if ( !start.hasPrev() )
            continue;
        if ( testExistance( start.a, start.prevA ) || testExistance( start.b, start.prevA ) ) // test if this edge exists
        {
            // fix multiple edge
            unsigned steps = ( start.b + unsigned( loop.size() ) - start.a ) % unsigned( loop.size() );
            getOptimalSteps( optimalStepsCache, ( start.a + 1 ) % loop.size(), steps, unsigned( loop.size() ), maxPolygonSubdivisions );
            optimalStepsCache.erase( std::remove_if( optimalStepsCache.begin(), optimalStepsCache.end(), [&] ( unsigned v )
            {
                return testExistance( start.a, v ) || testExistance( start.b, v ); // remove existing duplicates
            } ), optimalStepsCache.end() );
            if ( optimalStepsCache.empty() )
                return false;
            WeightedConn newPrev{ start.a,start.b,DBL_MAX,0 };
            getTriangulationWeights( topology, map, loop, metricRef, smoothBd, optimalStepsCache, newPrev ); // find better among steps
            if ( !newPrev.hasPrev() || !map[start.a][newPrev.prevA].hasPrev() || !map[start.prevA][newPrev.b].hasPrev() )
                return false;
            start.prevA = newPrev.prevA;
            mapPatch.emplace_back( MapPatchElement{ .a = start.a,.b = start.b,.newPrevA = start.prevA } );
        }
        auto distA = ( start.a - start.prevA + loop.size() ) % loop.size();
        auto distB = ( start.b - start.prevA + loop.size() ) % loop.size();
        if ( distA >= 2 && distA <= int( loop.size() ) - 2 )
            pushInQueue( start.a, start.prevA );
        if ( distB >= 2 && distB <= int( loop.size() ) - 2 )
            pushInQueue( start.prevA, start.b );
    }
    return true;
}

// add next candidate to queue
void processCandidate( const Mesh& mesh, const WeightedConn& current,
    std::priority_queue<WeightedConn>& queue, NewEdgesMap& map,
    const std::vector<EdgeId>& aEdgesMap,
    const std::vector<EdgeId>& bEdgesMap,
    const FillHoleMetric& metrics,
    bool addALoop )
{
    int nextA{ current.a };
    int nextB{ current.b };
    if ( addALoop )
        ++nextA;
    else
        ++nextB;

    // its not expected to have over 2 mln edges on one hole
    auto aEdgesMapSize = ( int )aEdgesMap.size();
    auto bEdgesMapSize = ( int )bEdgesMap.size();
    if ( nextA > aEdgesMapSize )
        return;
    if ( nextB > bEdgesMapSize )
        return;

    // do not allow full ring from one loop
    if ( ( nextA == aEdgesMapSize && nextB == 0 ) || ( nextB == bEdgesMapSize && nextA == 0 ) )
        return;

    auto& nextConn = map[nextA][nextB];

    VertId aVert = mesh.topology.org( aEdgesMap[current.a % aEdgesMapSize] );
    VertId bVert = mesh.topology.org( bEdgesMap[current.b % bEdgesMapSize] );
    if ( addALoop )
        std::swap( aVert, bVert );
    VertId cVert = addALoop ? mesh.topology.org( aEdgesMap[nextA % aEdgesMapSize] ) : mesh.topology.org( bEdgesMap[nextB % bEdgesMapSize] );

    VertId aOp{};
    if ( addALoop && mesh.topology.right( aEdgesMap[current.a % aEdgesMapSize] ) )
        aOp = mesh.topology.dest( mesh.topology.prev( aEdgesMap[current.a % aEdgesMapSize] ) );
    else if ( !addALoop && mesh.topology.right( bEdgesMap[nextB % bEdgesMapSize] ) )
        aOp = mesh.topology.dest( mesh.topology.prev( bEdgesMap[nextB % bEdgesMapSize] ) );
    VertId cOp{};
    if ( current.hasPrev() )
    {
        if ( ( current.prevA % aEdgesMapSize ) != ( current.a % aEdgesMapSize ) )
            cOp = mesh.topology.org( aEdgesMap[current.prevA % aEdgesMapSize] );
        else if ( ( current.prevB % bEdgesMapSize ) != ( current.b % bEdgesMapSize ) )
            cOp = mesh.topology.org( bEdgesMap[current.prevB % bEdgesMapSize] );
    }

    double sumMetric = current.weight;
    if ( metrics.triangleMetric )
    {
        auto triMetric = addALoop ?
            metrics.triangleMetric( bVert, aVert, cVert ):
            metrics.triangleMetric( aVert, bVert, cVert ); // always cw oriented
        sumMetric = metrics.combineMetric( sumMetric, triMetric );
    }

    if ( metrics.edgeMetric )
    {
        if ( cOp )
        {
            auto edgeACMetric = addALoop ?
                metrics.edgeMetric( bVert, aVert, cOp, cVert ) :
                metrics.edgeMetric( aVert, bVert, cOp, cVert ); // always cw oriented
            sumMetric = metrics.combineMetric( sumMetric, edgeACMetric );
        }
        if ( aOp )
        {
            auto edgeCBMetric = addALoop ?
                metrics.edgeMetric( cVert, bVert, aOp, aVert ) :
                metrics.edgeMetric( bVert, cVert, aOp, aVert ); // always cw oriented
            sumMetric = metrics.combineMetric( sumMetric, edgeCBMetric );
        }
    }

    if ( sumMetric >= nextConn.weight )
        return;
    // push to queue only if new variant is better as other way to this conn
    nextConn.a = nextA;
    nextConn.b = nextB;
    nextConn.weight = sumMetric;
    nextConn.prevA = current.a;
    nextConn.prevB = current.b;
    // this means that we reached end
    if ( nextA == aEdgesMapSize && nextB == bEdgesMapSize && metrics.edgeMetric )
    {
        // we need to spin back candidate to find first edge to count edge metric of last edge
        WeightedConn currentSpin;
        WeightedConn prevSpin = map[nextConn.prevA][nextConn.prevB];
        for ( ;;)
        {
            if ( prevSpin.hasPrev() )
            {
                currentSpin = prevSpin;
                prevSpin = map[currentSpin.prevA][currentSpin.prevB];
            }
            else
            {
                assert( currentSpin.a == 1 || currentSpin.b == 1 );
                double lastEdgeMetric = metrics.edgeMetric(
                    mesh.topology.org( aEdgesMap.front() ), mesh.topology.org( bEdgesMap.front() ), // 0-a,0-b
                    bVert, // if addLoopA - last a, otherwise last b
                    currentSpin.a == 1 ? mesh.topology.org( aEdgesMap[1] ) : mesh.topology.org( bEdgesMap[1] ) ); // 1-a or 1-b

                nextConn.weight = metrics.combineMetric( nextConn.weight, lastEdgeMetric );
                break;
            }
        }
    }
    // push last one too to terminate main loop
    queue.push( nextConn );
}

void buildCylinderBetweenTwoHoles( Mesh & mesh, EdgeId a0, EdgeId b0, const StitchHolesParams& params )
{
    MR_TIMER;
    MR_WRITER( mesh );

    auto newFace = [&] ()
    {
        auto res = mesh.topology.addFaceId();
        if ( params.outNewFaces )
            params.outNewFaces->autoResizeSet( res );
        return res;
    };
    if ( mesh.topology.left( a0 ) || mesh.topology.left( b0 ) )
    {
        assert( false );
        spdlog::error( "buildCylinderBetweenTwoHoles: edges do not represent holes" );
        return;
    }
    // stitch direction should be independent of input order
    if ( a0 < b0 )
        std::swap( a0, b0 );

    size_t aLoopEdgesCounter = 0;
    size_t bLoopEdgesCounter = 0;
    // find two closest points between two boundaries
    double minDistSq = DBL_MAX;
    EdgeId ac, bc; // origin vertices of these two edges are the closest
    EdgeId a = a0;
    do
    {
        Vector3d ap( mesh.orgPnt( a ) );
        EdgeId b = b0;
        do
        {
            Vector3d bp( mesh.orgPnt( b ) );
            double distSq = ( ap - bp ).lengthSq(); // compute lengthSq in double to avoid overflow even for very large floats in ap and bp
            if ( distSq < minDistSq )
            {
                minDistSq = distSq;
                ac = a;
                bc = b;
            }
            b = mesh.topology.prev( b.sym() );
            if ( aLoopEdgesCounter == 0 )
                ++bLoopEdgesCounter;
        } while ( b != b0 );
        a = mesh.topology.prev( a.sym() );
        ++aLoopEdgesCounter;
    } while ( a != a0 );

    if ( !ac || !bc )
    {
        assert( false );
        return;
    }

    // Fill EdgeMaps
    std::vector<EdgeId> aEdgeMap( aLoopEdgesCounter );
    std::vector<EdgeId> bEdgeMap( bLoopEdgesCounter );
    a = ac;
    for ( int i = 0; i < aLoopEdgesCounter; ++i )
    {
        aEdgeMap[i] = a;
        a = mesh.topology.prev( a.sym() );
    }
    EdgeId b = bc;
    for ( int i = 0; i < bLoopEdgesCounter; ++i )
    {
        bEdgeMap[i] = b;
        b = mesh.topology.next( b ).sym();
    }

    FillHoleMetric metrics = params.metric;
    if ( !metrics.edgeMetric && !metrics.triangleMetric )
        metrics = getComplexStitchMetric( mesh );
    if ( !metrics.combineMetric )
        metrics.combineMetric = [] ( double a, double b ) { return a + b; };

    // [0..aLoopEdgesCounter][0..bLoopEdgesCounter]
    // last one represents the same edge as first one, but reaching it means that algorithm has finished
    NewEdgesMap newEdgesMap( aLoopEdgesCounter + 1, std::vector<WeightedConn>( bLoopEdgesCounter + 1 ) );

    WeightedConn& firstWConn = newEdgesMap[0][0];
    firstWConn.a = 0; firstWConn.b = 0; firstWConn.weight = std::sqrt( minDistSq );
    std::priority_queue<WeightedConn> queue;
    WeightedConn current;
    queue.push( firstWConn );
    do
    {
        current = queue.top(); // cannot use std::move unfortunately since top() returns const reference
        queue.pop();
        if ( current.a == aEdgeMap.size() && current.b == bEdgeMap.size() )
            break; // if minimal one is [sizeA,sizeB] then terminate loop
        //add to queue next a variant and next b variant
        processCandidate( mesh, current, queue, newEdgesMap, aEdgeMap, bEdgeMap, metrics, true );
        processCandidate( mesh, current, queue, newEdgesMap, aEdgeMap, bEdgeMap, metrics, false );
    } while ( !queue.empty() );

    current = newEdgesMap.back().back();
    // connect two boundaries with the first edge
    EdgeId e1 = mesh.topology.makeEdge();
    mesh.topology.splice( ac, e1 );
    mesh.topology.splice( bc, e1.sym() );
    int prevA = ( int )aLoopEdgesCounter;
    // add next edges and make triangles
    // prev has prev, not to add same edge twice (first and last)
    while ( current.hasPrev() && newEdgesMap[current.prevA][current.prevB].hasPrev() )
    {
        current = newEdgesMap[current.prevA][current.prevB];
        EdgeId e2 = mesh.topology.makeEdge();
        // if curShared->b = prevB
        EdgeId sp = aEdgeMap[current.a % aLoopEdgesCounter];
        EdgeId symSp = mesh.topology.prev( e1.sym() );
        //else
        if ( current.a == prevA )
        {
            sp = e1;
            symSp = mesh.topology.prev( symSp.sym() );
        }
        mesh.topology.splice( sp, e2 );
        mesh.topology.splice( symSp, e2.sym() );
        mesh.topology.setLeft( e1, newFace() );
        prevA = current.a;
        e1 = e2;
    }
    // make last triangle without adding new edges
    mesh.topology.setLeft( e1, newFace() );
}

bool buildCylinderBetweenTwoHoles( Mesh & mesh, const StitchHolesParams& params )
{
    auto bdEdges = mesh.topology.findHoleRepresentiveEdges();
    if ( bdEdges.size() < 2 )
        return false;

    buildCylinderBetweenTwoHoles( mesh, bdEdges[0], bdEdges[1], params );
    return true;
}

// returns new edge connecting org(a) and org(b),
inline EdgeId makeNewEdge( MeshTopology & topology, EdgeId a, EdgeId b )
{
    EdgeId newEdge = topology.makeEdge();
    topology.splice( a, newEdge );
    topology.splice( b, newEdge.sym() );
    return newEdge;
}

void executeHoleFillPlan( Mesh & mesh, EdgeId a0, HoleFillPlan & plan, FaceBitSet * outNewFaces )
{
    [[maybe_unused]] const auto fsz0 = mesh.topology.faceSize();
    const FaceId f0 = mesh.topology.left( a0 );
    if ( plan.items.empty() )
    {
        if ( mesh.topology.isLeftTri( a0 ) )
        {
            assert( plan.numTris == 1 );
            if ( !f0 )
            {
                auto newFaceId = mesh.topology.addFaceId();
                if ( outNewFaces )
                    outNewFaces->autoResizeSet( newFaceId );
                mesh.topology.setLeft( a0, newFaceId );
            }
        }
        else
        {
            assert( plan.numTris >= 3 );
            fillHoleTrivially( mesh, a0, outNewFaces );
        }
    }
    else
    {
        if ( f0 )
            mesh.topology.setLeft( a0, {} );
        auto getEdge = [&]( int code )
        {
            if ( code >= 0 )
                return EdgeId( code );
            return EdgeId( plan.items[ -(code+1) ].edgeCode1 );
        };
        // make new edges
        for ( int i = 0; i < plan.items.size(); ++i )
        {
            EdgeId a = getEdge( plan.items[i].edgeCode1 );
            EdgeId b = getEdge( plan.items[i].edgeCode2 );
            EdgeId c = makeNewEdge( mesh.topology, a, b );
            plan.items[i].edgeCode1 = (int)c;
        }
        // restore old face
        if ( f0 )
        {
            auto e = EdgeId( plan.items[0].edgeCode1 );
            assert( !mesh.topology.left( e ) );
            assert( mesh.topology.isLeftTri( e ) );
            mesh.topology.setLeft( e, f0 );
        }
        // introduce new faces
        for ( int i = 0; i < plan.items.size(); ++i )
        {
            auto e = EdgeId( plan.items[i].edgeCode1 );
            for ( int j = 0; j < 2; ++j, e = e.sym() )
            {
                if ( mesh.topology.left( e ) )
                    continue;
                assert( mesh.topology.isLeftTri( e ) );
                auto f = mesh.topology.addFaceId();
                if ( outNewFaces )
                    outNewFaces->autoResizeSet( f );
                mesh.topology.setLeft( e, f );
            }
        }
    }
    [[maybe_unused]] const auto fsz = mesh.topology.faceSize();
    assert( plan.numTris == int( fsz - fsz0 + ( f0 ? 1 : 0 ) ) );
}

/// this class allows you to prepare fill plans for several holes with no new memory allocations on
/// second and subsequent calls
class HoleFillPlanner
{
public:
    HoleFillPlan run( const Mesh& mesh, EdgeId e, const FillHoleParams& params = {} );
    HoleFillPlan runPlanar( const Mesh& mesh, EdgeId e );

    bool parallelProcessing = true;

private:
    std::vector<EdgeId> edgeMap_;
    std::vector<std::vector<WeightedConn>> newEdgesMap_;
    tbb::enumerable_thread_specific<std::vector<unsigned>> optimalStepsCache_;
    MapPatch savedMapPatch_, cachedMapPatch_;
    std::queue<std::pair<WeightedConn, int>> newEdgesQueue_;
};

// Sub cubic complexity
HoleFillPlan HoleFillPlanner::run( const Mesh& mesh, EdgeId a0, const FillHoleParams& params )
{
    HoleFillPlan res;
    if ( params.stopBeforeBadTriangulation )
        *params.stopBeforeBadTriangulation = false;
    if ( params.maxPolygonSubdivisions < 2 )
    {
        assert( false );
        return res;
    }

    unsigned loopEdgesCounter = 0;
    EdgeId a = a0;
    do
    {
        a = mesh.topology.prev( a.sym() );
        ++loopEdgesCounter;
    } while ( a != a0 );

    if ( loopEdgesCounter <= 3 )
    {
        // no new edges, one triangle
        res.numTris = 1;
        return res;
    }

    // Fill EdgeMaps
    edgeMap_.clear();
    edgeMap_.reserve( loopEdgesCounter );
    a = a0;
    for ( unsigned i = 0; i < loopEdgesCounter; ++i )
    {
        edgeMap_.push_back( a );
        a = mesh.topology.prev( a.sym() );
    }

    // do not decrease the size not to deallocate nested vectors
    if ( newEdgesMap_.size() < loopEdgesCounter )
        newEdgesMap_.resize( loopEdgesCounter );
    for ( unsigned i = 0; i < loopEdgesCounter; ++i )
    {
        newEdgesMap_[i].clear();
        newEdgesMap_[i].resize( loopEdgesCounter, { -1,-1,0.0,0 } );
    }

    FillHoleMetric metrics = params.metric;
    if ( !metrics.edgeMetric && !metrics.triangleMetric )
        metrics = getCircumscribedMetric( mesh );
    if ( !metrics.combineMetric )
        metrics.combineMetric = [] ( double a, double b ) { return a + b; };

    //fill all table not queue
    constexpr unsigned stepStart = 2;
    const unsigned stepEnd = loopEdgesCounter - 2;
    for ( auto steps = stepStart; steps <= stepEnd; ++steps )
    {
        auto work = [&]( unsigned i, std::vector<unsigned>& optimalSteps )
        {
            const auto cIndex = ( i + steps ) % loopEdgesCounter;
            EdgeId aCur = edgeMap_[i];
            EdgeId cCur = edgeMap_[cIndex];
            WeightedConn& current = newEdgesMap_[i][cIndex];
            current = { int( i ),int( cIndex ), DBL_MAX,0 };
            if ( params.multipleEdgesResolveMode != FillHoleParams::MultipleEdgesResolveMode::None &&
                sameEdgeExists( mesh.topology, aCur, cCur ) )
                return;
            getOptimalSteps( optimalSteps, ( i + 1 ) % loopEdgesCounter, steps, loopEdgesCounter, params.maxPolygonSubdivisions );
            getTriangulationWeights( mesh.topology, newEdgesMap_, edgeMap_, metrics, params.smoothBd, optimalSteps, current ); // find better among steps
        };
        if ( parallelProcessing )
        {
            ParallelFor( unsigned( 0 ), loopEdgesCounter, optimalStepsCache_, [&]( unsigned i, std::vector<unsigned>& optimalSteps )
            {
                work( i, optimalSteps );
            } );
        }
        else
        {
            auto & optimalSteps = optimalStepsCache_.local();
            for ( unsigned i = 0; i < loopEdgesCounter; ++i )
                work( i, optimalSteps );
        }

    }
    // find minimum triangulation
    savedMapPatch_.clear();
    cachedMapPatch_.clear();
    WeightedConn finConn{-1,-1,DBL_MAX};
    for ( unsigned i = 0; i < loopEdgesCounter; ++i )
    {
        const auto cIndex = ( i + stepStart ) % loopEdgesCounter;
        double weight = metrics.combineMetric( newEdgesMap_[i][cIndex].weight, newEdgesMap_[cIndex][i].weight );
        if ( metrics.edgeMetric )
        {
            VertId leftVert;
            if ( newEdgesMap_[i][cIndex].hasPrev() )
                leftVert = mesh.topology.org( edgeMap_[newEdgesMap_[i][cIndex].prevA] );
            else if ( params.smoothBd && mesh.topology.right( edgeMap_[i] ) )
                leftVert = mesh.topology.dest( mesh.topology.prev( edgeMap_[i] ) );

            VertId rightVert;
            if ( newEdgesMap_[cIndex][i].hasPrev() )
                rightVert = mesh.topology.org( edgeMap_[newEdgesMap_[cIndex][i].prevA] );
            else if ( params.smoothBd && mesh.topology.right( edgeMap_[cIndex] ) )
                rightVert = mesh.topology.dest( mesh.topology.prev( edgeMap_[cIndex] ) );

            if ( leftVert && rightVert )
            {
                auto lastEdgeMetric = metrics.edgeMetric( mesh.topology.org( edgeMap_[i] ), mesh.topology.org( edgeMap_[cIndex] ), leftVert, rightVert );
                weight = metrics.combineMetric( weight, lastEdgeMetric );
            }
        }
        if ( weight < finConn.weight &&
            ( params.multipleEdgesResolveMode != FillHoleParams::MultipleEdgesResolveMode::Strong || // try to fix multiple if needed
                removeMultipleEdgesFromTriangulation( mesh.topology, newEdgesMap_, edgeMap_, metrics, params.smoothBd, newEdgesMap_[cIndex][i], params.maxPolygonSubdivisions, cachedMapPatch_ ) ) )
        {
            savedMapPatch_ = cachedMapPatch_;
            finConn = newEdgesMap_[cIndex][i];
            finConn.weight = weight;
        }
    }

    if ( params.stopBeforeBadTriangulation )
    {
        if ( finConn.a == -1 || finConn.b == -1 || finConn.weight > BadTriangulationMetric )
        {
            *params.stopBeforeBadTriangulation = true;
            return res;
        }
    }

    if ( finConn.a == -1 || finConn.b == -1 )
    {
        // "trivial" fill
        res.numTris = loopEdgesCounter;
        return res;
    }

    if ( params.multipleEdgesResolveMode == FillHoleParams::MultipleEdgesResolveMode::Strong && !savedMapPatch_.empty() )
        for ( const auto& [patchA, patchB, patchPrevA] : savedMapPatch_ )
            newEdgesMap_[patchA][patchB].prevA = patchPrevA;

    // queue for adding new edges (not to make tree like recursive logic)
    WeightedConn fictiveLastConn( finConn.a, ( finConn.b + 1 ) % loopEdgesCounter, 0.0 );
    fictiveLastConn.prevA = finConn.b;
    assert( newEdgesQueue_.empty() );
    newEdgesQueue_.push( {fictiveLastConn,(int)edgeMap_[fictiveLastConn.b]} );
    std::pair<WeightedConn, int> curConn;
    while ( !newEdgesQueue_.empty() )
    {
        curConn = std::move( newEdgesQueue_.front() );
        newEdgesQueue_.pop();

        auto distA = ( curConn.first.a - curConn.first.prevA + loopEdgesCounter ) % loopEdgesCounter;
        auto distB = ( curConn.first.b - curConn.first.prevA + loopEdgesCounter ) % loopEdgesCounter;

        if ( distA >= 2 && distA <= loopEdgesCounter - 2 )
        {
            auto newEdgeCode = -int( res.items.size() + 1 );
            res.items.push_back( { (int)edgeMap_[curConn.first.prevA], (int)edgeMap_[curConn.first.a] } );
            newEdgesQueue_.push( { newEdgesMap_[curConn.first.a][curConn.first.prevA], newEdgeCode } );
        }

        if ( distB >= 2 && distB <= loopEdgesCounter - 2 )
        {
            auto newEdgeCode = -int( res.items.size() + 1 );
            res.items.push_back( { (int)curConn.second, (int)edgeMap_[curConn.first.prevA] } );
            newEdgesQueue_.push( { newEdgesMap_[curConn.first.prevA][curConn.first.b], newEdgeCode } );
        }

        ++res.numTris;
    }
    return res;
}

HoleFillPlan HoleFillPlanner::runPlanar( const Mesh& mesh, EdgeId e )
{
    bool stopOnBad{ false };
    FillHoleParams params;
    params.metric = getPlaneNormalizedFillMetric( mesh, e );
    params.stopBeforeBadTriangulation = &stopOnBad;

    auto res = run( mesh, e, params );
    if ( stopOnBad ) // triangulation cannot be good if we fall in this `if`, so let it create degenerated faces
        res = run( mesh, e, { getMinAreaMetric( mesh ) } );
    return res;
}

HoleFillPlan getHoleFillPlan( const Mesh& mesh, EdgeId e, const FillHoleParams& params )
{
    return HoleFillPlanner{}.run( mesh, e, params );
}

std::vector<HoleFillPlan> getHoleFillPlans( const Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges, const FillHoleParams& params )
{
    MR_TIMER;
    std::vector<HoleFillPlan> fillPlans( holeRepresentativeEdges.size() );
    tbb::enumerable_thread_specific<HoleFillPlanner> threadData_;
    ParallelFor( holeRepresentativeEdges, threadData_, [&]( size_t i, HoleFillPlanner& planner )
    {
        planner.parallelProcessing = false; // to prevent run() from calling this lambda for a different i-index
        fillPlans[i] = planner.run( mesh, holeRepresentativeEdges[i], params );
    } );
    return fillPlans;
}

HoleFillPlan getPlanarHoleFillPlan( const Mesh& mesh, EdgeId e )
{
    return HoleFillPlanner{}.runPlanar( mesh, e );
}

std::vector<HoleFillPlan> getPlanarHoleFillPlans( const Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges )
{
    MR_TIMER;
    std::vector<HoleFillPlan> fillPlans( holeRepresentativeEdges.size() );
    tbb::enumerable_thread_specific<HoleFillPlanner> threadData_;
    ParallelFor( holeRepresentativeEdges, threadData_, [&]( size_t i, HoleFillPlanner& planner )
    {
        planner.parallelProcessing = false; // to prevent run() from calling this lambda for a different i-index
        fillPlans[i] = planner.runPlanar( mesh, holeRepresentativeEdges[i] );
    } );
    return fillPlans;
}

bool isHoleBd( const MeshTopology & topology, const EdgeLoop & loop )
{
    if ( loop.empty() )
        return false;

    const EdgeId a0 = loop.front();
    EdgeId a = a0;
    int n = 0;
    for (;;)
    {
        if ( topology.left( a ) )
            return false;
        a = topology.prev( a.sym() );
        ++n;
        if ( a == a0 )
            break;
        if ( n >= loop.size() )
            return false;
        if ( a != loop[n] )
            return false;
    }
    return n == loop.size();
}

void fillHole( Mesh& mesh, EdgeId a0, const FillHoleParams& params )
{
    MR_TIMER;
    MR_WRITER( mesh );
    assert( !mesh.topology.left( a0 ) );
    if ( mesh.topology.left( a0 ) )
        return;

    unsigned loopEdgesCounter = 0;
    EdgeId a = a0;
    do
    {
        a = mesh.topology.prev( a.sym() );
        ++loopEdgesCounter;
    } while ( a != a0 );

    if ( loopEdgesCounter < 2 )
    {
        // loop hole
        assert( false );
        return;
    }

    if ( params.makeDegenerateBand )
    {
        a = a0 = makeDegenerateBandAroundHole( mesh, a0, params.outNewFaces );
        for ( unsigned i = 0; i < loopEdgesCounter; ++i )
            a = mesh.topology.prev( a.sym() );
    }

    if ( loopEdgesCounter == 2 )
    {
        EdgeId a1 = mesh.topology.next( a0 );
        EdgeId a2 = mesh.topology.prev( a1.sym() );
        mesh.topology.splice( a0, a1 );
        mesh.topology.splice( a2, a1.sym() );
        assert( mesh.topology.isLoneEdge( a1 ) );
        return;
    }

    auto plan = getHoleFillPlan( mesh, a0, params );
    if ( params.stopBeforeBadTriangulation && *params.stopBeforeBadTriangulation )
        return;

    executeHoleFillPlan( mesh, a0, plan, params.outNewFaces );
}

void fillHoles( Mesh& mesh, const std::vector<EdgeId> & as, const FillHoleParams& params )
{
    MR_TIMER;

    // TODO: parallel getHoleFillPlan

    for ( auto a : as )
        fillHole( mesh, a, params );
}

VertId fillHoleTrivially( Mesh& mesh, EdgeId a, FaceBitSet * outNewFaces /*= nullptr */ )
{
    MR_WRITER( mesh );
    const FaceId f0 = mesh.topology.left( a );
    if ( f0 )
        mesh.topology.setLeft( a, {} );

    auto addFaceId = [&]()
    {
        auto res = mesh.topology.addFaceId();
        if ( outNewFaces )
            outNewFaces->autoResizeSet( res );
        return res;
    };

    Vector3d sum;
    int holeDegree = 0;
    for ( EdgeId e : leftRing( mesh.topology, a ) )
    {
        sum += Vector3d{ mesh.orgPnt( e ) };
        ++holeDegree;
    }
    const Vector3f centerPos{ sum / double( holeDegree ) };
    const VertId centerVert = mesh.addPoint( centerPos );

    // from boundary org( a ) to center point
    const EdgeId e0 = mesh.topology.makeEdge();
    mesh.topology.splice( a, e0 );

    EdgeId elast = e0;
    auto bdi = mesh.topology.prev( a.sym() );
    for ( int i = 1; i < holeDegree; ++i )
    {
        auto bdi1 = mesh.topology.prev( bdi.sym() );
        EdgeId ei = mesh.topology.makeEdge();
        mesh.topology.splice( bdi, ei );
        mesh.topology.splice( elast.sym(), ei.sym() );
        assert( mesh.topology.isLeftTri( ei ) );
        mesh.topology.setLeft( ei, addFaceId() );
        elast = ei;
        bdi = bdi1;
    }
    // and last face
    assert( mesh.topology.isLeftTri( e0 ) );
    mesh.topology.setLeft( e0, f0 ? f0 : addFaceId() );

    mesh.topology.setOrg( e0.sym(), centerVert );

    return centerVert;
}

EdgeId extendHole( Mesh& mesh, EdgeId a, std::function<Vector3f(const Vector3f &)> getVertPos, FaceBitSet * outNewFaces /*= nullptr */ )
{
    MR_TIMER;
    MR_WRITER( mesh );
    assert( !mesh.topology.left( a ) );

    auto addFaceId = [&]()
    {
        auto res = mesh.topology.addFaceId();
        if ( outNewFaces )
            outNewFaces->autoResizeSet( res );
        return res;
    };

    EdgeId e0, ei;
    e0 = ei = mesh.topology.makeEdge();
    mesh.topology.splice( a, e0 );
    mesh.topology.setOrg( e0.sym(), mesh.addPoint( getVertPos( mesh.orgPnt( a ) ) ) );
    EdgeId res;
    for ( EdgeId ai = a; ai != e0; )
    {
        EdgeId ai1 = mesh.topology.prev( ai.sym() );
        EdgeId em = mesh.topology.makeEdge();
        mesh.topology.splice( ei.sym(), em.sym() );
        mesh.topology.splice( ai1, em );
        assert( mesh.topology.isLeftTri( em ) );
        mesh.topology.setLeft( em, addFaceId() );

        EdgeId ei1;
        if ( ai1 == e0 )
            ei1 = e0;
        else
        {
            ei1 = mesh.topology.makeEdge();
            mesh.topology.splice( ai1, ei1 );
            mesh.topology.setOrg( ei1.sym(), mesh.addPoint(getVertPos( mesh.destPnt( ai ) ) ) );
        }
        EdgeId opp = mesh.topology.makeEdge();
        mesh.topology.splice( em.sym(), opp );
        mesh.topology.splice( opp.sym(), mesh.topology.prev( ei1.sym() ) );
        assert( mesh.topology.isLeftTri( ei1 ) );
        mesh.topology.setLeft( ei1, addFaceId() );
        if ( a == ai )
            res = opp;
        ei = ei1;
        ai = ai1;
    }
    return res;
}

EdgeId extendHole( Mesh& mesh, EdgeId a, const Plane3f & plane, FaceBitSet * outNewFaces )
{
    return extendHole( mesh, a,
        [plane]( const Vector3f & p ) { return plane.project( p ); },
        outNewFaces );
}

std::vector<EdgeId> extendAllHoles( Mesh& mesh, const Plane3f & plane, FaceBitSet * outNewFaces )
{
    MR_TIMER;
    auto borders = mesh.topology.findHoleRepresentiveEdges();

    for ( auto& border : borders )
        border = extendHole( mesh, border, plane, outNewFaces );

    return borders;
}

EdgeId buildBottom( Mesh& mesh, EdgeId a, Vector3f dir, float holeExtension, FaceBitSet* outNewFaces /*= nullptr */ )
{
    dir = dir.normalized();
    float min = FLT_MAX;
    VertId minVert;
    for ( auto next : leftRing( mesh.topology, a ) )
    {
        VertId vid = mesh.topology.org( next );
        float dist = dot( mesh.points[vid], dir );
        if ( dist < min )
        {
            min = dist;
            minVert = vid;
        }
    }

    return extendHole( mesh, a, Plane3f::fromDirAndPt( dir, mesh.points[minVert] - holeExtension * dir ), outNewFaces );
}

EdgeId makeDegenerateBandAroundHole( Mesh& mesh, EdgeId a, FaceBitSet * outNewFaces )
{
    return extendHole( mesh, a,
        []( const Vector3f & p ) { return p; },
        outNewFaces );
}

MakeBridgeResult makeQuadBridge( MeshTopology & topology, EdgeId a, EdgeId b, FaceBitSet * outNewFaces )
{
    assert( !topology.left( a ) );
    assert( !topology.left( b ) );
    MakeBridgeResult res;
    if ( a == b )
        return res;
    bool swapped = false;
    if ( topology.prev( b.sym() ) == a )
    {
        swapped = true;
        std::swap( a, b );
    }
    if ( topology.prev( a.sym() ) == b )
    {
        if ( !topology.isLeftTri( a ) )
        {
            const auto bDest = topology.dest( b );
            for ( auto e : orgRing0( topology, a ) )
            {
                if ( topology.dest( e ) == bDest )
                {
                    // there is an edge between org(a) and dest(b), so if create another one, then multiple edges appear
                    return res;
                }
            }

            // specific case of neighboring edges
            if ( !topology.isLeftTri( a ) )
            {
                auto e = topology.makeEdge();
                topology.splice( a, e );
                topology.splice( topology.prev( b.sym() ), e.sym() );
                if ( swapped )
                    res.nb = e;
                else
                    res.na = e;
            }
        }
        auto f = topology.addFaceId();
        topology.setLeft( a, f );
        ++res.newFaces;
        if ( outNewFaces )
            outNewFaces->autoResizeSet( f );
        assert( !res.na || ( !swapped && topology.fromSameOriginRing( a, res.na ) && !topology.left( res.na ) ) );
        assert( !res.nb || (  swapped && topology.fromSameOriginRing( a, res.nb ) && !topology.left( res.nb ) ) );
        return res;
    }
    assert( !swapped );

    // general case

    const auto bOrg = topology.org( b );
    const auto bDest = topology.dest( b );
    for ( auto e : orgRing( topology, a ) )
    {
        const auto eDest = topology.dest( e );
        if ( eDest == bOrg || eDest == bDest )
        {
            // there is an edge between org(a) and ( org(b) or dest(b) ), so if create another one, then multiple edges appear
            return res;
        }
    }
    for ( auto e : orgRing( topology, a.sym() ) )
    {
        const auto eDest = topology.dest( e );
        if ( eDest == bOrg || eDest == bDest )
        {
            // there is an edge between dest(a) and ( org(b) or dest(b) ), so if create another one, then multiple edges appear
            return res;
        }
    }

    auto c = topology.makeEdge();
    auto e = topology.makeEdge();
    topology.splice( topology.prev( a.sym() ), c );
    topology.splice( a, e.sym() );
    res.na = e.sym();
    topology.splice( topology.prev( b.sym() ), e );
    topology.splice( b, c.sym() );
    res.nb = c.sym();
    assert( topology.isLeftQuad( a ) );

    auto fa = topology.addFaceId();
    topology.setLeft( a, fa );
    ++res.newFaces;
    assert( res.na && topology.fromSameOriginRing( a, res.na ) && !topology.left( res.na ) );
    assert( res.nb && topology.fromSameOriginRing( b, res.nb ) && !topology.left( res.nb ) );
    if ( outNewFaces )
        outNewFaces->autoResizeSet( fa );
    return res;
}

void splitQuad( MeshTopology & topology, EdgeId a, FaceBitSet * outNewFaces )
{
    assert( topology.isLeftQuad( a ) );
    assert( topology.left( a ) );
    auto d = topology.makeEdge();
    topology.splice( topology.prev( a.sym() ), d );
    topology.splice( topology.next( a ).sym(), d.sym() );
    assert( topology.isLeftTri( d ) );
    assert( topology.isLeftTri( d.sym() ) );
    assert( topology.left( d ) );
    assert( !topology.left( d.sym() ) );
    auto f = topology.addFaceId();
    topology.setLeft( d.sym(), f );
    if ( outNewFaces )
        outNewFaces->autoResizeSet( f );
}

MakeBridgeResult makeBridge( MeshTopology & topology, EdgeId a, EdgeId b, FaceBitSet * outNewFaces )
{
    auto res = makeQuadBridge( topology, a, b, outNewFaces );
    if ( res.na && res.nb )
    {
        assert( res.newFaces == 1 );
        splitQuad( topology, a, outNewFaces );
        ++res.newFaces;
    }
    return res;
}

MakeBridgeResult makeSmoothBridge( Mesh & mesh, EdgeId a, EdgeId b, float samplingStep, FaceBitSet * outNewFaces )
{
    MR_TIMER;
    MakeBridgeResult res = makeQuadBridge( mesh.topology, a, b, outNewFaces );
    if ( !res.na && !res.nb )
        return res;

    // build spline starting in the middle of edge (a) and ending in the middle of edge (b),
    // with tangents at the ends inside existing triangles
    const Vector3f centerA = mesh.edgeCenter( a );
    const Vector3f centerB = mesh.edgeCenter( b );
    const Vector3f tangentA = mesh.leftTangent( a.sym() );
    const Vector3f tangentB = mesh.leftTangent( b.sym() );
    const Vector3f dirA = mesh.edgeVector( a ).normalized();
    const Vector3f dirB = mesh.edgeVector( b.sym() ).normalized();

    const float tangentStep = 0.99f * samplingStep; // to avoid splitting of tangents

    Contour3f normals{ dirA, dirA, dirB, dirB };
    auto marked = makeSpline( Contour3f{ centerA + tangentStep * tangentA, centerA, centerB, centerB + tangentStep * tangentB },
        { .samplingStep = samplingStep, .controlStability = 10, .iterations = 3, .normals = &normals } );
    assert( normals.size() == marked.contour.size() );

    const EdgeId ca = res.na;
    // split the only bridge's triangle or quadrangle on segments according to the spline
    const int midPoints = (int)normals.size() - 4;
    if ( midPoints > 0 )
    {
        const auto lenA = mesh.edgeLength( a );
        const auto lenB = mesh.edgeLength( b );
        for ( int i = 0; i < midPoints; ++i )
        {
            const auto u = float( i + 1 ) / ( midPoints + 1 );
            const auto len = ( 1 - u ) * lenA + u * lenB;
            if ( ca )
            {
                // split the first boundary of the bridge
                const auto p = marked.contour[i + 2] - 0.5f * len * normals[i + 2];
                const auto e = mesh.splitEdge( ca, p, outNewFaces );
                ++res.newFaces;
                assert( mesh.topology.isLeftTri( e.sym() ) );
                if ( i == 0 )
                    res.na = e;
            }
            if ( res.nb )
            {
                // split the second boundary of the bridge
                const auto p = marked.contour[i + 2] + 0.5f * len * normals[i + 2];
                [[maybe_unused]] const auto e = mesh.splitEdge( res.nb.sym(), p, outNewFaces );
                assert( mesh.topology.isLeftTri( e ) );
                ++res.newFaces;
            }
        }
    }
    if ( ca && mesh.topology.isLeftQuad( ca.sym() ) )
    {
        // split the last quadrangle;
        // mesh.topology.prev( ca ) below to have the same diagonal as in above quadrangles
        splitQuad( mesh.topology, mesh.topology.prev( ca ), outNewFaces );
        ++res.newFaces;
    }

    assert( !res.na || ( mesh.topology.fromSameOriginRing( a, res.na ) && !mesh.topology.left( res.na ) ) );
    assert( !res.nb || ( mesh.topology.fromSameOriginRing( b, res.nb ) && !mesh.topology.left( res.nb ) ) );
    return res;
}

EdgeId makeBridgeEdge( MeshTopology & topology, EdgeId a, EdgeId b )
{
    assert( !topology.left( a ) );
    assert( !topology.left( b ) );
    if ( topology.fromSameOriginRing( a, b ) )
    {
        // org(a) and org(b) is the same vertex, so we cannot create a loop edge
        return {};
    }

    const auto bOrg = topology.org( b );
    for ( auto e : orgRing( topology, a ) )
    {
        if ( topology.dest( e ) == bOrg )
        {
            // there is an edge between org(a) and org(b), so if create another one, then multiple edges appear
            return {};
        }
    }

    auto res = topology.makeEdge();
    topology.splice( a, res );
    topology.splice( b, res.sym() );
    return res;
}

TEST( MRMesh, buildCylinderBetweenTwoHoles )
{
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 3_v, 4_v, 5_v }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );
    EXPECT_EQ( mesh.topology.numValidVerts(), 6 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 2 );

    mesh.points.emplace_back( 0.f, 0.f, 0.f ); // VertId{0}
    mesh.points.emplace_back( 1.f, 0.f, 0.f ); // VertId{1}
    mesh.points.emplace_back( 0.f, 1.f, 0.f ); // VertId{2}
    mesh.points.emplace_back( 0.f, 0.f, 1.f ); // VertId{3}
    mesh.points.emplace_back( 1.f, 0.f, 1.f ); // VertId{4}
    mesh.points.emplace_back( 0.f, 1.f, 1.f ); // VertId{5}
    EXPECT_EQ( mesh.points.size(), 6 );

    auto bdEdges = mesh.topology.findHoleRepresentiveEdges();
    EXPECT_EQ( bdEdges.size(), 2 );
    EXPECT_FALSE( mesh.topology.left( bdEdges[0] ).valid() );
    EXPECT_FALSE( mesh.topology.left( bdEdges[1] ).valid() );

    FaceBitSet newFaces;
    StitchHolesParams params;
    auto fsz0 = mesh.topology.faceSize();
    params.outNewFaces = &newFaces;
    buildCylinderBetweenTwoHoles( mesh, bdEdges[0], bdEdges[1], params );
    auto numNewFaces = mesh.topology.faceSize() - fsz0;

    EXPECT_EQ( mesh.topology.numValidVerts(), 6 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 8 );
    EXPECT_EQ( mesh.points.size(), 6 );
    EXPECT_EQ( numNewFaces, 6 );
    EXPECT_EQ( newFaces.count(), 6 );
    EXPECT_EQ( newFaces.size(), 8 );

    bdEdges = mesh.topology.findHoleRepresentiveEdges();
    EXPECT_EQ( bdEdges.size(), 0 );
}

TEST( MRMesh, makeBridge )
{
    MeshTopology topology;
    auto a = topology.makeEdge();
    topology.setOrg( a, topology.addVertId() );
    topology.setOrg( a.sym(), topology.addVertId() );
    auto b = topology.makeEdge();
    topology.setOrg( b, topology.addVertId() );
    topology.setOrg( b.sym(), topology.addVertId() );
    EXPECT_EQ( topology.numValidFaces(), 0 );
    FaceBitSet fbs;
    auto bridgeRes = makeBridge( topology, a, b, &fbs );
    EXPECT_TRUE( bridgeRes );
    EXPECT_EQ( bridgeRes.newFaces, 2 );
    EXPECT_TRUE( bridgeRes.na );
    EXPECT_EQ( topology.org( a ), topology.org( bridgeRes.na ) );
    EXPECT_TRUE( topology.left( a ) );
    EXPECT_FALSE( topology.left( bridgeRes.na ) );
    EXPECT_TRUE( bridgeRes.nb );
    EXPECT_EQ( topology.org( b ), topology.org( bridgeRes.nb ) );
    EXPECT_TRUE( topology.left( b ) );
    EXPECT_FALSE( topology.left( bridgeRes.nb ) );
    EXPECT_EQ( fbs.count(), 2 );
    EXPECT_EQ( topology.numValidVerts(), 4 );
    EXPECT_EQ( topology.numValidFaces(), 2 );
    EXPECT_EQ( topology.edgeSize(), 5 * 2 );

    topology = MeshTopology();
    a = topology.makeEdge();
    topology.setOrg( a, topology.addVertId() );
    topology.setOrg( a.sym(), topology.addVertId() );
    b = topology.makeEdge();
    topology.splice( a.sym(), b );
    topology.setOrg( b.sym(), topology.addVertId() );
    EXPECT_EQ( topology.numValidFaces(), 0 );
    fbs.reset();
    bridgeRes = makeBridge( topology, a, b, &fbs );
    EXPECT_TRUE( bridgeRes );
    EXPECT_EQ( bridgeRes.newFaces, 1 );
    EXPECT_TRUE( bridgeRes.na );
    EXPECT_EQ( topology.org( a ), topology.org( bridgeRes.na ) );
    EXPECT_TRUE( topology.left( a ) );
    EXPECT_FALSE( topology.left( bridgeRes.na ) );
    EXPECT_FALSE( bridgeRes.nb );
    EXPECT_TRUE( topology.left( b ) );
    EXPECT_EQ( fbs.count(), 1 );
    EXPECT_EQ( topology.numValidVerts(), 3 );
    EXPECT_EQ( topology.numValidFaces(), 1 );
    EXPECT_EQ( topology.edgeSize(), 3 * 2 );
}

TEST( MRMesh, makeBridgeEdge )
{
    MeshTopology topology;
    auto a = topology.makeEdge();
    topology.setOrg( a, topology.addVertId() );
    topology.setOrg( a.sym(), topology.addVertId() );
    auto b = topology.makeEdge();
    topology.setOrg( b, topology.addVertId() );
    topology.setOrg( b.sym(), topology.addVertId() );
    auto x = makeBridgeEdge( topology, a, b );
    EXPECT_TRUE( topology.fromSameOriginRing( a, x ) );
    EXPECT_TRUE( topology.fromSameOriginRing( b, x.sym() ) );
    EXPECT_EQ( topology.edgeSize(), 3 * 2 );

    x = makeBridgeEdge( topology, a, b );
    EXPECT_FALSE( x.valid() );
}

} //namespace MR
