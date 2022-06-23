#include "MRMeshFillHole.h"
#include "MRMesh.h"
#include "MRBitSet.h"
#include "MRVector3.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRPlane3.h"
#include "MRMeshBuilder.h"
#include "MRMeshDelone.h"
#include "MRHash.h"
#include <parallel_hashmap/phmap.h>
#include "MRPch/MRTBB.h"
#include <queue>
#include "MRGTest.h"
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
    WeightedConn( int _a, int _b, double _weight ) :
        a{_a}, b{_b}, weight{_weight}    {}
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

constexpr int maxStepsNum = 20;
void getOptimalSteps( std::vector<unsigned>& optimalSteps, unsigned start, unsigned steps, unsigned loopSize )
{
    optimalSteps.resize(0);
    --steps;
    if ( steps <= maxStepsNum )
    {
        for ( unsigned i = 0; i < steps; ++i )
            optimalSteps.push_back( ( start + i ) % loopSize );
        return;
    }

    for ( int i = 0; i < ( maxStepsNum / 4 ); ++i )
        optimalSteps.push_back( ( start + i ) % loopSize );


    auto bigStep = ( steps - ( maxStepsNum / 2 ) ) / ( maxStepsNum / 2 );
    auto numBigSteps = ( maxStepsNum / 2 );
    if ( bigStep < 2 )
    {
        bigStep = 2;
        numBigSteps = ( maxStepsNum / 4 );
    }
    auto bigStepHalf = bigStep / 2;
    const auto newStart = start + bigStepHalf + ( maxStepsNum / 4 ) - 1;
    for ( int i = 0; i < numBigSteps; ++i )
        optimalSteps.push_back( ( newStart + i * bigStep ) % loopSize );

    for ( int i = ( maxStepsNum / 4 ) -1; i >=0 ; --i )
        optimalSteps.push_back( ( start + steps - i - 1 ) % loopSize );
}

// finds best candidate among all given steps
void getTriangulationWeights( const MeshTopology& topology, const NewEdgesMap& map, const EdgePath& loop,
    const FillHoleMetric& metricRef,
    const std::vector<unsigned>& optimalStepsCache, WeightedConn& processedConn )
{
    for ( unsigned s = 0; s < optimalStepsCache.size(); ++s )
    {
        auto v = optimalStepsCache[s];
        const auto& abConn = map[processedConn.a][v];
        const auto& bcConn = map[v][processedConn.b];
        double weight = abConn.weight + bcConn.weight;
        if ( weight > processedConn.weight )
            continue;

        VertId aVert = topology.org( loop[processedConn.a] );
        VertId bVert = topology.org( loop[processedConn.b] );

        if ( aVert == bVert )
            continue;

        //weight += metricTriangle( mesh, aCur, edgeMap[v], cCur );
        weight += metricRef.getTriangleMetric(
            aVert, topology.org( loop[v] ), bVert,
            !bcConn.hasPrev() ? topology.dest( topology.prev( loop[v] ) ) : topology.org( loop[bcConn.prevA] ),
            !abConn.hasPrev() ? topology.dest( topology.prev( loop[processedConn.a] ) ) : topology.org( loop[abConn.prevA] )
        );
        if ( weight < processedConn.weight )
        {
            processedConn.weight = weight;
            processedConn.prevA = v; // In this case prevA describes chosen triangulation
        }
    }
}

// this function go backward by given triangulation and tries to fix multiple edges
// return false if triangulation has multiple edges that cannot be fixed
bool removeMultipleEdgesFromTriangulation( const MeshTopology& topology, NewEdgesMap& map, const EdgePath& loop,  const FillHoleMetric& metricRef,
    WeightedConn start )
{
    MR_TIMER;

    phmap::flat_hash_set<std::pair<VertId, VertId>> edgesInTriangulation;
    auto testExistance = [&] ( VertId a, VertId b )->bool
    {
        if ( a > b )
            std::swap( a, b );
        auto it = edgesInTriangulation.find( { a,b } );
        return it != edgesInTriangulation.end();
    };
    std::vector<unsigned> optimalStepsCache( maxStepsNum );
    std::queue<WeightedConn> newEdgesQueue;
    newEdgesQueue.push( start );
    while ( !newEdgesQueue.empty() )
    {
        start = std::move( newEdgesQueue.front() );
        newEdgesQueue.pop();

        VertId aV, bV;
        aV = topology.org( loop[start.a] );
        bV = topology.org( loop[start.b] );
        if ( aV > bV )
            std::swap( aV, bV );
        edgesInTriangulation.insert( { aV,bV } ); // add edge to map

        if ( start.hasPrev() )
        {
            VertId prevV = topology.org( loop[start.prevA] );
            if ( testExistance( aV, prevV ) || testExistance( bV, prevV ) ) // test if this edge exists
            {
                // fix multiple edge 
                unsigned steps = ( start.b + unsigned( loop.size() ) - start.a ) % unsigned( loop.size() );
                getOptimalSteps( optimalStepsCache, ( start.a + 1 ) % loop.size(), steps, unsigned( loop.size() ) );
                optimalStepsCache.erase( std::remove_if( optimalStepsCache.begin(), optimalStepsCache.end(), [&] ( unsigned v )
                {
                    VertId vV = topology.org( loop[v] );
                    return testExistance( aV, vV ) || testExistance( bV, vV ); // remove existing duplicates
                } ) );
                if ( optimalStepsCache.empty() )
                    return false;
                WeightedConn newPrev{ start.a,start.b,DBL_MAX };
                getTriangulationWeights( topology, map, loop, metricRef, optimalStepsCache, newPrev ); // find better among steps
                if ( !newPrev.hasPrev() )
                    return false;
                start.prevA = newPrev.prevA;
                map[start.a][start.b].prevA = start.prevA;
            }
            auto distA = ( start.a - start.prevA + loop.size() ) % loop.size();
            auto distB = ( start.b - start.prevA + loop.size() ) % loop.size();
            if ( distA >= 2 && distA <= int( loop.size() ) - 2 )
                newEdgesQueue.push( map[start.a][start.prevA] );
            if ( distB >= 2 && distB <= int( loop.size() ) - 2 )
                newEdgesQueue.push( map[start.prevA][start.b] );
        }
    }
    return true;
}

// add next candidate to queue
// return true if reaches end of loop
// otherwise returns false
bool processCandidate( const Mesh& mesh, const WeightedConn& current,
    std::priority_queue<WeightedConn>& queue, NewEdgesMap& map,
    const std::vector<EdgeId>& aEdgesMap,
    const std::vector<EdgeId>& bEdgesMap,
    const FillHoleMetric& metric,
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
        return false;
    if ( nextB > bEdgesMapSize )
        return false;

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

    // do not allow full ring from one loop
    if ( ( nextA == aEdgesMapSize && nextB == 0 ) || ( nextB == bEdgesMapSize && nextA == 0 ) )
        return false;

    // Metric is sum of length so we add new to previous one
    double sumMetric = current.weight + metric.getTriangleMetric( aVert, bVert, cVert, aOp, cOp );
    if ( sumMetric >= nextConn.weight )
        return false;
    // push to queue only if new variant is better as other way to this conn
    nextConn.a = nextA;
    nextConn.b = nextB;
    nextConn.weight = sumMetric;
    nextConn.prevA = current.a;
    nextConn.prevB = current.b;
    // this means that we reached end
    if ( nextA == aEdgesMapSize && nextB == bEdgesMapSize )
    {
        if ( !metric.hasEdgeMetric )
            return true;
        // we need to spin back candidate to find first edge to count edge metric of last edge
        WeightedConn currentSpin;
        WeightedConn prevSpin = map[nextConn.prevA % aEdgesMapSize][nextConn.prevB % bEdgesMapSize];
        for ( ;;)
        {
            if ( prevSpin.hasPrev() )
            {
                currentSpin = prevSpin;
                prevSpin = map[currentSpin.prevA % aEdgesMapSize][currentSpin.prevB % bEdgesMapSize];
            }
            else
            {
                assert( currentSpin.a == 1 || currentSpin.b == 1 );
                nextConn.weight += metric.getEdgeMetric( mesh.topology.org( aEdgesMap.front() ), mesh.topology.org( bEdgesMap.front() ),
                    cOp, currentSpin.a == 1 ? mesh.topology.org( aEdgesMap[1] ) : mesh.topology.org( bEdgesMap[1] ) );
                break;
            }

        }
    }
    else
    {
        queue.push( nextConn );
    }

    return false;
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

    size_t aLoopEdgesCounter = 0;
    size_t bLoopEdgesCounter = 0;
    // find two closest points between two boundaries
    double minDistSq = DBL_MAX;
    EdgeId ac, bc; // origin vertices of these two edges are the closest
    EdgeId a = a0;
    do
    {
        auto ap = mesh.orgPnt( a );
        EdgeId b = b0;
        do
        {
            auto bp = mesh.orgPnt( b );
            double distSq = ( ap - bp ).lengthSq();
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


    if ( !params.metric )
        const_cast< StitchHolesParams& >( params ).metric = std::make_unique<ComplexStitchMetric>( mesh );

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
        //add to queue next a variant and next b variant
        if ( processCandidate( mesh, current, queue, newEdgesMap, aEdgeMap, bEdgeMap, *params.metric, true ) ||
            processCandidate( mesh, current, queue, newEdgesMap, aEdgeMap, bEdgeMap, *params.metric, false ) )
            break;
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

// Sub cubic complexity
void fillHole( Mesh& mesh, EdgeId a0, const FillHoleParams& params )
{
    MR_TIMER;
    MR_WRITER( mesh );
    if ( mesh.topology.left( a0 ) )
        return;
    assert( !mesh.topology.left( a0 ) );
    auto newFace = [&]()
    {
        auto res = mesh.topology.addFaceId();
        if ( params.outNewFaces )
            params.outNewFaces->autoResizeSet( res );
        return res;
    };

    unsigned loopEdgesCounter = 0;
    EdgeId a = a0;
    do
    {
        a = mesh.topology.prev( a.sym() );
        ++loopEdgesCounter;
    } while ( a != a0 );

    if ( loopEdgesCounter < 3 )
        return;

    if ( loopEdgesCounter == 3)
    {
        mesh.topology.setLeft( a, newFace() );
        return;
    }
    // Fill EdgeMaps
    std::vector<EdgeId> edgeMap( loopEdgesCounter );
    a = a0;
    for ( unsigned i = 0; i < loopEdgesCounter; ++i )
    {
        edgeMap[i] = a;
        a = mesh.topology.prev( a.sym() );
    }

    NewEdgesMap newEdgesMap( loopEdgesCounter, std::vector<WeightedConn>( loopEdgesCounter, {-1,-1,0.0} ) );

    if ( !params.metric )
        const_cast< FillHoleParams& >( params ).metric = std::make_unique<CircumscribedFillMetric>( mesh );

    const FillHoleMetric& metricRef = *params.metric;

    //fill all table not queue
    constexpr unsigned stepStart = 2;
    const unsigned stepEnd = loopEdgesCounter - 2;
    for ( auto steps = stepStart; steps <= stepEnd; ++steps )
    {
        tbb::parallel_for( tbb::blocked_range<unsigned>( 0, loopEdgesCounter, 15 ), [&]( const tbb::blocked_range<unsigned>& range )
        {
            std::vector<unsigned> optimalStepsCache;
            optimalStepsCache.resize( maxStepsNum );
            for ( unsigned i = range.begin(); i < range.end(); ++i )
            {
                const auto cIndex = ( i + steps ) % loopEdgesCounter;
                EdgeId aCur = edgeMap[i];
                EdgeId cCur = edgeMap[cIndex];
                WeightedConn& current = newEdgesMap[i][cIndex];
                current = {int( i ),int( cIndex ), DBL_MAX};
                if ( params.multipleEdgesResolveMode != FillHoleParams::MultipleEdgesResolveMode::None &&
                    sameEdgeExists( mesh.topology, aCur, cCur ) )
                    continue;
                getOptimalSteps( optimalStepsCache, ( i + 1 ) % loopEdgesCounter, steps, loopEdgesCounter );
                getTriangulationWeights( mesh.topology, newEdgesMap, edgeMap, metricRef, optimalStepsCache, current ); // find better among steps
            }
        });
    }
    // find minimum triangulation
    WeightedConn finConn{-1,-1,DBL_MAX};
    for ( unsigned i = 0; i < loopEdgesCounter; ++i )
    {
        const auto cIndex = ( i + stepStart ) % loopEdgesCounter;
        double weight = newEdgesMap[i][cIndex].weight + newEdgesMap[cIndex][i].weight +
            metricRef.getEdgeMetric( 
                mesh.topology.org( edgeMap[i] ), mesh.topology.org( edgeMap[cIndex] ),
                !newEdgesMap[i][cIndex].hasPrev() ? mesh.topology.dest( mesh.topology.prev( edgeMap[i] ) ) : mesh.topology.org( edgeMap[newEdgesMap[i][cIndex].prevA] ),
                !newEdgesMap[cIndex][i].hasPrev() ? mesh.topology.dest( mesh.topology.prev( edgeMap[cIndex] ) ) : mesh.topology.org( edgeMap[newEdgesMap[cIndex][i].prevA] )
            );

        if ( weight < finConn.weight &&
            ( params.multipleEdgesResolveMode != FillHoleParams::MultipleEdgesResolveMode::Strong || // try to fix multiple if needed
                removeMultipleEdgesFromTriangulation( mesh.topology, newEdgesMap, edgeMap, metricRef, newEdgesMap[cIndex][i] ) ) )
        {
            finConn = newEdgesMap[cIndex][i];
            finConn.weight = weight;
        }
    }

    if ( params.makeDegenerateBand )
    {
        a = a0 = makeDegenerateBandAroundHole( mesh, a0, params.outNewFaces );
        for ( unsigned i = 0; i < loopEdgesCounter; ++i )
        {
            edgeMap[i] = a;
            a = mesh.topology.prev( a.sym() );
        }
    }

    if ( finConn.a == -1 || finConn.b == -1 )
    {
        fillHoleTrivially( mesh, a0, params.outNewFaces );
        return;
    }


    // queue for adding new edges (not to make tree like recursive logic)
    WeightedConn fictiveLastConn( finConn.a, ( finConn.b + 1 ) % loopEdgesCounter, 0.0 );
    fictiveLastConn.prevA = finConn.b;
    std::queue<std::pair<WeightedConn, EdgeId>> newEdgesQueue;
    newEdgesQueue.push( {fictiveLastConn,edgeMap[fictiveLastConn.b]} );
    std::pair<WeightedConn, EdgeId> curConn;
    while ( !newEdgesQueue.empty() )
    {
        curConn = std::move( newEdgesQueue.front() );
        newEdgesQueue.pop();

        auto distA = ( curConn.first.a - curConn.first.prevA + loopEdgesCounter ) % loopEdgesCounter;
        auto distB = ( curConn.first.b - curConn.first.prevA + loopEdgesCounter ) % loopEdgesCounter;

        if ( distA >= 2 && distA <= loopEdgesCounter - 2 )
        {
            EdgeId newEdge = mesh.topology.makeEdge();
            mesh.topology.splice( edgeMap[curConn.first.prevA], newEdge );
            mesh.topology.splice( edgeMap[curConn.first.a], newEdge.sym() );
            newEdgesQueue.push( {newEdgesMap[curConn.first.a][curConn.first.prevA],newEdge} );
        }

        if ( distB >= 2 && distB <= loopEdgesCounter - 2 )
        {
            EdgeId newEdge = mesh.topology.makeEdge();
            mesh.topology.splice( curConn.second, newEdge );
            mesh.topology.splice( edgeMap[curConn.first.prevA], newEdge.sym() );
            newEdgesQueue.push( {newEdgesMap[curConn.first.prevA][curConn.first.b],newEdge} );
        }

        mesh.topology.setLeft( curConn.second, newFace() );
    }
}

VertId fillHoleTrivially( Mesh& mesh, EdgeId a, FaceBitSet * outNewFaces /*= nullptr */ )
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

    Vector3d sum;
    std::vector<EdgeId> bdEdges;
    for ( EdgeId e : leftRing( mesh.topology, a ) )
    {
        bdEdges.push_back( e );
        sum += Vector3d{ mesh.orgPnt( e ) };
    }
    const Vector3f centerPos{ sum / double( bdEdges.size() ) };
    const VertId centerVert = mesh.addPoint( centerPos );

    // from boundary org( bdEdges[0] ) to center point
    const EdgeId e0 = mesh.topology.makeEdge();
    mesh.topology.splice( bdEdges[0], e0 );

    EdgeId elast = e0;
    for ( int i = 1; i < bdEdges.size(); ++i )
    {
        EdgeId ei = mesh.topology.makeEdge();
        mesh.topology.splice( bdEdges[i], ei );
        mesh.topology.splice( elast.sym(), ei.sym() );
        assert( mesh.topology.isLeftTri( ei ) );
        mesh.topology.setLeft( ei, addFaceId() );
        elast = ei;
    }
    // and last face
    assert( mesh.topology.isLeftTri( e0 ) );
    mesh.topology.setLeft( e0, addFaceId() );

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

bool makeBridge( MeshTopology & topology, EdgeId a, EdgeId b, FaceBitSet * outNewFaces )
{
    assert( !topology.left( a ) );
    assert( !topology.left( b ) );
    if ( a == b )
    {
        return false;
    }
    if ( topology.prev( b.sym() ) == a )
        std::swap( a, b );
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
                    return false;
                }
            }

            // specific case of neighboring edges
            if ( !topology.isLeftTri( a ) )
            {
                auto e = topology.makeEdge();
                topology.splice( a, e );
                topology.splice( topology.prev( b.sym() ), e.sym() );
            }
        }
        auto f = topology.addFaceId();
        topology.setLeft( a, f );
        if ( outNewFaces )
            outNewFaces->autoResizeSet( f );
        return true;
    }

    // general case

    const auto bOrg = topology.org( b );
    const auto bDest = topology.dest( b );
    for ( auto e : orgRing( topology, a ) )
    {
        const auto eDest = topology.dest( e );
        if ( eDest == bOrg || eDest == bDest )
        {
            // there is an edge between org(a) and ( org(b) or dest(b) ), so if create another one, then multiple edges appear
            return false;
        }
    }
    for ( auto e : orgRing( topology, a.sym() ) )
    {
        const auto eDest = topology.dest( e );
        if ( eDest == bOrg || eDest == bDest )
        {
            // there is an edge between dest(a) and ( org(b) or dest(b) ), so if create another one, then multiple edges appear
            return false;
        }
    }

    auto c = topology.makeEdge();
    auto d = topology.makeEdge();
    auto e = topology.makeEdge();
    topology.splice( topology.prev( a.sym() ), c );
    topology.splice( c, d );
    topology.splice( a, e.sym() );
    topology.splice( topology.prev( b.sym() ), e );
    topology.splice( e, d.sym() );
    topology.splice( b, c.sym() );
    assert( topology.isLeftTri( a ) );
    assert( topology.isLeftTri( b ) );

    auto fa = topology.addFaceId();
    topology.setLeft( a, fa );
    auto fb = topology.addFaceId();
    topology.setLeft( b, fb );
    if ( outNewFaces )
    {
        outNewFaces->autoResizeSet( fa );
        outNewFaces->autoResizeSet( fb );
    }
    return true;
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
    std::vector<VertId> v{
        VertId{0}, VertId{1}, VertId{2},
        VertId{3}, VertId{4}, VertId{5}
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromVertexTriples( v );
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
    params.outNewFaces = &newFaces;
    buildCylinderBetweenTwoHoles( mesh, bdEdges[0], bdEdges[1], params );

    EXPECT_EQ( mesh.topology.numValidVerts(), 6 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 8 );
    EXPECT_EQ( mesh.points.size(), 6 );
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
    FaceBitSet fbs;
    EXPECT_TRUE( makeBridge( topology, a, b, &fbs ) );
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
    fbs.reset();
    makeBridge( topology, a, b, &fbs );
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
