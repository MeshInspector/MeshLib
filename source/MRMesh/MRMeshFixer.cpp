#include "MRMeshFixer.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
#include "MRTriMath.h"
#include "MRParallelFor.h"
#include "MRLine3.h"
#include "MRMeshIntersect.h"
#include "MRBox.h"
#include "MRRegionBoundary.h"
#include "MRExpandShrink.h"
#include "MRMeshDecimate.h"
#include "MRMeshSubdivide.h"
#include "MREdgePaths.h"
#include "MRFillHoleNicely.h"

namespace MR
{

// given a vertex, returns two edges with the origin in this vertex consecutive in the vertex ring without left faces both;
// both edges may be the same if there is only one edge without left face;
// or both edges can be invalid if all vertex edges have left face
static EdgePair getTwoSeqNoLeftAtVertex( const MeshTopology & m, VertId a )
{
    EdgeId e0 = m.edgeWithOrg( a );
    if ( !e0.valid() )
        return {}; //invalid vertex

    // find first hole edge
    EdgeId eh = e0;
    for (;;)
    {
        if ( !m.left( eh ).valid() )
            break;
        eh = m.next( eh );
        if ( eh == e0 )
            return {}; // no single hole near a
    }

    // find second hole edge
    for ( EdgeId e = m.next( eh ); e != e0; e = m.next( e ) )
    {
        if ( !m.left( e ).valid() )
            return { eh, e }; // another hole near a
    }

    return { eh, eh };
}

int duplicateMultiHoleVertices( Mesh & mesh )
{
    int duplicates = 0;
    const auto lastVert = mesh.topology.lastValidVert();
    for ( VertId v{0}; v <= lastVert; ++v )
    {
        auto ee = getTwoSeqNoLeftAtVertex( mesh.topology, v );
        if ( ee.first == ee.second )
            continue;

        EdgeId e1 = ee.first;
        EdgeId e0 = e1;
        while ( mesh.topology.right( e0 ).valid() )
            e0 = mesh.topology.prev( e0 );

        // unsplice [e0, e1] and create new vertex for it
        mesh.topology.splice( mesh.topology.prev( e0 ), e1 );
        assert( !mesh.topology.org( e0 ).valid() );

        auto vDup = mesh.addPoint( mesh.points[v] );
        mesh.topology.setOrg( e0, vDup );

        ++duplicates;
        --v;
    }

    return duplicates;
}

Expected<std::vector<MultipleEdge>> findMultipleEdges( const MeshTopology& topology, ProgressCallback cb )
{
    MR_TIMER;
    tbb::enumerable_thread_specific<std::vector<MultipleEdge>> threadData;
    const VertId lastValidVert = topology.lastValidVert();

    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };
    std::atomic<size_t> numDone{ 0 };
    tbb::parallel_for( tbb::blocked_range<size_t>( size_t{ 0 },  size_t( lastValidVert ) + 1 ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        auto & tls = threadData.local();
        std::vector<VertId> neis;
        for ( VertId v = VertId( range.begin() ); v < VertId( range.end() ); ++v )
        {
            if ( cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            if ( !topology.hasVert( v ) )
                continue;
            neis.clear();
            for ( auto e : orgRing( topology, v ) )
            {
                auto nv = topology.dest( e );
                if ( nv > v )
                    neis.push_back( nv );
            }
            std::sort( neis.begin(), neis.end() );
            auto it = neis.begin();
            for (;;)
            {
                it = std::adjacent_find( it, neis.end() );
                if ( it == neis.end() )
                    break;
                auto nv = *it;
                tls.emplace_back( v, nv );
                assert( nv == *( it + 1 ) );
                ++++it;
                while ( it != neis.end() && *it == nv )
                    ++it;
                if ( it == neis.end() )
                    break;
            }
        }

        if ( cb )
            numDone += range.size();

        if ( cb && std::this_thread::get_id() == mainThreadId )
        {
            if ( !cb( float( numDone ) / float( lastValidVert + 1 ) ) )
                keepGoing.store( false, std::memory_order_relaxed );
        }
    } );

    if ( !keepGoing.load( std::memory_order_relaxed ) || ( cb && !cb( 1.0f ) ) )
        return unexpectedOperationCanceled();

    std::vector<MultipleEdge> res;
    for ( const auto & ns : threadData )
        res.insert( res.end(), ns.begin(), ns.end() );
    // sort the result to make it independent of mesh distribution among threads
    std::sort( res.begin(), res.end() );

    return res;
}

Expected<void> fixMeshDegeneracies( Mesh& mesh, const FixMeshDegeneraciesParams& params )
{
    MR_TIMER;
    int maxSteps = 1;
    if ( params.mode == FixMeshDegeneraciesParams::Mode::Remesh )
        maxSteps = 2;
    else if ( params.mode == FixMeshDegeneraciesParams::Mode::RemeshPatch )
        maxSteps = 3;

    auto prepareRegion = [&] ( auto cb )->Expected<FaceBitSet>
    {
        auto dfres = findDegenerateFaces( { mesh,params.region }, params.criticalTriAspectRatio, subprogress( cb, 0.0f, 0.5f ) );
        if ( !dfres.has_value() )
            return unexpected( dfres.error() );
        auto seres = findShortEdges( { mesh,params.region }, params.tinyEdgeLength, subprogress( cb, 0.5f, 1.0f ) );
        if ( !seres.has_value() )
            return unexpected( seres.error() );
        if ( dfres->none() && seres->none() )
            return {}; // nothing to fix
        FaceBitSet tempRegion = *dfres | getIncidentFaces( mesh.topology, *seres );
        expand( mesh.topology, tempRegion, 3 );
        tempRegion &= mesh.topology.getFaceIds( params.region );
        return tempRegion;
    };

    // START DECIMATION PART
    auto sbd = subprogress( params.cb, 0.0f, 1.0f / float( maxSteps ) );
    auto regRes = prepareRegion( subprogress( sbd, 0.0f, 0.2f ) );
    if ( !regRes.has_value() )
        return unexpected( regRes.error() );
    if ( regRes->none() )
        return {}; // nothing to fix
    if ( !reportProgress( sbd, 0.25f ) )
        return unexpectedOperationCanceled();

    DecimateSettings dsettings
    {
        .strategy = DecimateStrategy::ShortestEdgeFirst,
        .maxError = params.maxDeviation,
        .criticalTriAspectRatio = maxSteps > 1 ? FLT_MAX : params.criticalTriAspectRatio, // no need to bypass checks in decimation if subdivision is on
        .tinyEdgeLength = params.tinyEdgeLength,
        .stabilizer = params.stabilizer,
        .optimizeVertexPos = false, // this decreases probability of normal inversion near mesh degenerations
        .region = &*regRes,
        .maxAngleChange = params.maxAngleChange,
        .progressCallback = subprogress( sbd, 0.25f,  1.0f )
    };

    auto res = decimateMesh( mesh, dsettings );

    if ( params.region )
    {
        // validate region
        *params.region |= *regRes;
        *params.region &= mesh.topology.getValidFaces();
    }

    if ( res.cancelled )
        return unexpectedOperationCanceled();

    if ( maxSteps == 1 )
        return {}; // other steps are disabled

    // START SUBDIVISION PART
    auto sbs = subprogress( params.cb, 1.0f / float( maxSteps ), 2.0f / float( maxSteps ) );
    regRes = prepareRegion( subprogress( sbs, 0.0f, 0.2f ) );
    if ( !regRes.has_value() )
        return unexpected( regRes.error() );
    if ( regRes->none() )
        return {}; // nothing to fix
    if ( !reportProgress( sbs, 0.25f ) )
        return unexpectedOperationCanceled();

    SubdivideSettings ssettings{
        .maxEdgeLen = 1e3f * params.tinyEdgeLength,
        .maxEdgeSplits = int( mesh.topology.undirectedEdgeSize() ), // 2 * int( region.count() ),
        .maxDeviationAfterFlip = params.maxDeviation, // 0.1 * tolerance
        .maxAngleChangeAfterFlip = params.maxAngleChange,
        .criticalAspectRatioFlip = params.criticalTriAspectRatio, // questionable - may lead to exceeding beyond tolerance, but if set FLT_MAX, may lead to more degeneracies
        .region = params.region,
        .maxTriAspectRatio = params.criticalTriAspectRatio,
        .progressCallback = subprogress( sbs, 0.25f, 1.0f )
    };
    subdivideMesh( mesh, ssettings );

    if ( !reportProgress( sbs, 1.f ) )
        return unexpectedOperationCanceled();

    if ( maxSteps == 2 )
        return {}; // other steps are disabled

    // START PATCH STEP
    auto sbp = subprogress( params.cb, 2.0f / float( maxSteps ), 3.0f / float( maxSteps ) );
    regRes = prepareRegion( subprogress( sbp, 0.0f, 0.2f ) );
    if ( !regRes.has_value() )
        return unexpected( regRes.error() );
    if ( regRes->none() )
        return {}; // nothing to fix
    if ( !reportProgress( sbp, 0.25f ) )
        return unexpectedOperationCanceled();

    auto boundaryEdges = delRegionKeepBd( mesh, *regRes );

    auto sb = subprogress( sbp, 0.25f, 1.0f );
    for ( int i = 0; i < boundaryEdges.size(); ++i )
    {
        const auto& boundaryEdge = boundaryEdges[i];
        if ( boundaryEdge.empty() )
            continue;

        const auto len = calcPathLength( boundaryEdge, mesh );
        const auto avgLen = len / boundaryEdge.size();
        FillHoleNicelySettings settings
        {
            .triangulateParams =
            {
                .metric = getUniversalMetric( mesh ),
                .multipleEdgesResolveMode = FillHoleParams::MultipleEdgesResolveMode::Strong,
            },
            .maxEdgeLen = float( avgLen ) * 1.5f,
            .maxEdgeSplits = 20'000,
            .smoothCurvature = true,
            .edgeWeights = EdgeWeights::Unit // use unit weights to avoid potential laplacian degeneration (which leads to nan coords)
        };

        for ( auto e : boundaryEdge )
        {
            if ( mesh.topology.left( e ) )
                continue;
            auto newFaces = fillHoleNicely( mesh, e, settings );
            if ( params.region )
                *params.region |= newFaces;
        }
        if ( !reportProgress( sb, ( i + 1.f ) / boundaryEdges.size() ) )
            return unexpectedOperationCanceled();
    }
    if ( params.region )
        *params.region &= mesh.topology.getValidFaces();
    return {};
}

VertBitSet findNRingVerts( const MeshTopology& topology, int n, const VertBitSet* region /*= nullptr */ )
{
    const auto& zone = topology.getVertIds( region );
    VertBitSet result( zone.size() );
    BitSetParallelFor( zone, [&] ( VertId v )
    {
        int counter = 0;
        for ( auto e : orgRing( topology, v ) )
        {
            if ( !topology.left( e ) )
                return;
            ++counter;
            if ( counter > n )
                return;
        }
        if ( counter < n )
            return;
        assert( counter == n );
        result.set( v );
    } );
    return result;
}

Expected<FaceBitSet> findDisorientedFaces( const Mesh& mesh, const FindDisorientationParams& params )
{
    MR_TIMER;
    auto disorientedFaces = mesh.topology.getValidFaces();

    Mesh cpyMesh;
    const Mesh* targetMesh{ &mesh };
    EdgeBitSet outHoles;
    if ( params.virtualFillHoles && mesh.topology.findNumHoles( &outHoles ) > 0 )
    {
        cpyMesh = mesh;
        targetMesh = &cpyMesh;
        auto sb = subprogress( params.cb, 0.0f, 0.5f );
        int i = 0;
        int num = int( outHoles.count() );
        auto metric = getMinAreaMetric( mesh );
        for ( auto e : outHoles )
        {
            ++i;
            fillHole( cpyMesh, e, { .metric = metric } ); // use simplest filling
            if ( !reportProgress( sb, float( i ) / float( num ) ) )
                return unexpectedOperationCanceled();
        }
    }

    auto sb = subprogress( params.cb, targetMesh == &mesh ? 0.0f : 0.5f, 1.0f );

    auto keepGoing = BitSetParallelFor( mesh.topology.getValidFaces(), [&] ( FaceId f )
    {
        auto normal = Vector3d( mesh.normal( f ) );
        auto triCenter = Vector3d( mesh.triCenter( f ) );
        int counter = 0;
        auto interPred = [f, &counter] ( const MeshIntersectionResult& res )->bool
        {
            if ( res.proj.face != f ) // TODO: we should also try grouping intersections, to ignore too close ones (by some epsilon), to filter several layered areas
                ++counter;
            return true;
        };
        rayMeshIntersectAll( *targetMesh, Line3d( triCenter, normal ), interPred );
        bool pValid = counter % 2 == 0;
        auto pCounter = counter;
        bool nValid = true;
        int nCounter = INT_MAX;
        bool resValid = pValid;
        if ( params.mode != FindDisorientationParams::RayMode::Positive )
        {
            counter = 0;
            rayMeshIntersectAll( *targetMesh, Line3d( triCenter, -normal ), interPred );
            nValid = counter % 2 == 1;
            nCounter = counter - 1; // ideal face has 0-pCounter and 1-nCounter: so we decrement nCounter for fair compare

            resValid = pValid && nValid;
            if ( params.mode == FindDisorientationParams::RayMode::Shallowest && pValid != nValid )
            {
                if ( pCounter == nCounter )
                    resValid = true;
                else if ( nCounter < pCounter )
                    resValid = nValid;
            }
        }

        if ( resValid )
            disorientedFaces.reset( f );
    }, sb );

    if ( !keepGoing )
        return unexpectedOperationCanceled();

    return disorientedFaces;
}

void fixMultipleEdges( Mesh & mesh, const std::vector<MultipleEdge> & multipleEdges )
{
    if ( multipleEdges.empty() )
        return;
    MR_TIMER;
    MR_WRITER( mesh )

    for ( const auto & mE : multipleEdges )
    {
        int num = 0;
        for ( auto e : orgRing( mesh.topology, mE.first ) )
        {
            if ( mesh.topology.dest( e ) != mE.second )
                continue;
            if ( num++ == 0 )
                continue; // skip the first edge in the group
            mesh.splitEdge( e.sym() );
        }
        assert( num > 1 ); //it was really multiply connected pair of vertices
    }
}

void fixMultipleEdges( Mesh & mesh )
{
    fixMultipleEdges( mesh, findMultipleEdges( mesh.topology ).value() );
}

Expected<FaceBitSet> findDegenerateFaces( const MeshPart& mp, float criticalAspectRatio, ProgressCallback cb )
{
    MR_TIMER;
    FaceBitSet res( mp.mesh.topology.faceSize() );
    auto completed = BitSetParallelFor( mp.mesh.topology.getFaceIds( mp.region ), [&] ( FaceId f )
    {
        if ( !mp.mesh.topology.hasFace( f ) )
            return;
        if ( mp.mesh.triangleAspectRatio( f ) >= criticalAspectRatio )
            res.set( f );
    }, cb );

    if ( !completed )
        return unexpectedOperationCanceled();

    return res;
}

Expected<UndirectedEdgeBitSet> findShortEdges( const MeshPart& mp, float criticalLength, ProgressCallback cb )
{
    MR_TIMER;
    const auto criticalLengthSq = sqr( criticalLength );
    UndirectedEdgeBitSet res( mp.mesh.topology.undirectedEdgeSize() );
    auto completed = BitSetParallelForAll( res, [&] ( UndirectedEdgeId ue )
    {
        if ( !mp.mesh.topology.isInnerOrBdEdge( ue, mp.region ) )
            return;
        if ( mp.mesh.edgeLengthSq( ue ) <= criticalLengthSq )
            res.set( ue );
    }, cb );

    if ( !completed )
        return unexpectedOperationCanceled();

    return res;
}

bool isEdgeBetweenDoubleTris( const MeshTopology& topology, EdgeId e )
{
    return topology.next( e.sym() ) == topology.prev( e.sym() ) &&
        topology.isLeftTri( e ) && topology.isLeftTri( e.sym() );
}

EdgeId eliminateDoubleTris( MeshTopology& topology, EdgeId e, FaceBitSet * region )
{
    const auto ex = topology.next( e.sym() );
    const EdgeId ep = topology.prev( e );
    const EdgeId en = topology.next( e );
    if ( ex != topology.prev( e.sym() ) || ep == en || !topology.isLeftTri( e ) || !topology.isLeftTri( e.sym() ) )
        return {};
    // left( e ) and right( e ) are double triangles
    if ( auto f = topology.left( e ) )
    {
        if ( region )
            region->reset( f );
        topology.setLeft( e, {} );
    }
    if ( auto f = topology.left( e.sym() ) )
    {
        if ( region )
            region->reset( f );
        topology.setLeft( e.sym(), {} );
    }
    topology.setOrg( e.sym(), {} );
    topology.splice( e.sym(), ex );
    topology.splice( ep, e );
    assert( topology.isLoneEdge( e ) );
    topology.splice( en.sym(), ex.sym() );
    assert( topology.isLoneEdge( ex ) );
    topology.splice( ep, en );
    topology.splice( topology.prev( en.sym() ), en.sym() );
    assert( topology.isLoneEdge( en ) );
    return ep;
}

void eliminateDoubleTrisAround( MeshTopology & topology, VertId v, FaceBitSet * region )
{
    EdgeId e = topology.edgeWithOrg( v );
    EdgeId e0 = e;
    for (;;)
    {
        if ( auto ep = eliminateDoubleTris( topology, e, region ) )
            e0 = e = ep;
        else
        {
            e = topology.next( e );
            if ( e == e0 )
                break; // full ring has been inspected
            continue;
        }
    }
}

bool isDegree3Dest( const MeshTopology& topology, EdgeId e )
{
    const EdgeId ex = topology.next( e.sym() );
    const EdgeId ey = topology.prev( e.sym() );
    return topology.next( ex ) == ey &&
        topology.isLeftTri( e ) && topology.isLeftTri( e.sym() ) && topology.isLeftTri( ex );
}

EdgeId eliminateDegree3Dest( MeshTopology& topology, EdgeId e, FaceBitSet * region )
{
    const EdgeId ex = topology.next( e.sym() );
    const EdgeId ey = topology.prev( e.sym() );
    const EdgeId ep = topology.prev( e );
    const EdgeId en = topology.next( e );
    if ( ep == en || topology.next( ex ) != ey ||
        !topology.isLeftTri( e ) || !topology.isLeftTri( e.sym() ) || !topology.isLeftTri( ex ) )
        return {};
    topology.flipEdge( ex );
    auto res = eliminateDoubleTris( topology, e, region );
    assert( res == ex );
    return res;
}

int eliminateDegree3Vertices( MeshTopology& topology, VertBitSet & region, FaceBitSet * fs )
{
    MR_TIMER;
    auto candidates = region;
    int res = 0;
    for (;;)
    {
        const int x = res;
        for ( auto v : candidates )
        {
            candidates.reset( v );
            const auto e0 = topology.edgeWithOrg( v );
            if ( !isDegree3Dest( topology, e0.sym() ) )
                continue;
            ++res;
            region.reset( v );
            for ( auto e : orgRing( topology, e0 ) )
                if ( auto vn = topology.dest( e ); region.test( vn ) )
                    candidates.autoResizeSet( vn );
            [[maybe_unused]] auto ep = eliminateDegree3Dest( topology, e0.sym(), fs );
            assert( ep );
        }
        if ( res == x )
            break;
    }
    return res;
}

EdgeId isVertexRepeatedOnHoleBd( const MeshTopology& topology, VertId v )
{
    for ( EdgeId e0 : orgRing( topology, v ) )
    {
        if ( topology.left( e0 ) )
            continue;
        // not very optional in case of many boundary edges, but it shall be rare
        for ( EdgeId e1 : orgRing0( topology, e0 ) )
        {
            if ( topology.left( e1 ) )
                continue;
            if ( topology.fromSameLeftRing( e0, e1 ) )
                return e0;
        }
    }
    return {};
}

VertBitSet findRepeatedVertsOnHoleBd( const MeshTopology& topology )
{
    MR_TIMER;
    const auto holeRepresEdges = topology.findHoleRepresentiveEdges();

    VertBitSet res;
    if ( holeRepresEdges.empty() )
        return res;

    struct ThreadData
    {
        explicit ThreadData( size_t vertSize ) : repeatedVerts( vertSize ), currHole( vertSize ) {}

        VertBitSet repeatedVerts;
        VertBitSet currHole;
    };

    tbb::enumerable_thread_specific<ThreadData> tls( topology.vertSize() );
    ParallelFor( holeRepresEdges, tls, [&]( size_t i, ThreadData & threadData )
    {
        const auto e0 = holeRepresEdges[i];
        for ( auto e : leftRing( topology, e0 ) )
        {
            auto v = topology.org( e );
            if ( threadData.currHole.test_set( v ) )
                threadData.repeatedVerts.set( v );
        }
        for ( auto e : leftRing( topology, e0 ) )
        {
            auto v = topology.org( e );
            threadData.currHole.reset( v );
        }
    } );

    for ( const auto & threadData : tls )
        res |= threadData.repeatedVerts;
    return res;
}

/// adds in complicatingFaces the faces not from the wedge with largest angle of faces connected by edges incident to given vertex
static void findHoleComplicatingFaces( const Mesh & mesh, VertId v, std::vector<FaceId> & complicatingFaces )
{
    EdgeId bd;
    float bdAngle = -1;

    auto angle = [&]( EdgeId e )
    {
        assert( !mesh.topology.left( e ) );
        float res = 0;
        while ( mesh.topology.right( e ) )
        {
            auto d1 = mesh.edgeVector( e );
            auto d0 = mesh.edgeVector( e = mesh.topology.prev( e ) );
            res += MR::angle( d0, d1 );
        }
        return res;
    };

    auto report = [&]( EdgeId e )
    {
        assert( !mesh.topology.left( e ) );
        while ( auto r = mesh.topology.right( e ) )
        {
            complicatingFaces.push_back( r );
            e = mesh.topology.prev( e );
        }
    };

    for ( EdgeId e : orgRing( mesh.topology, v ) )
    {
        if ( mesh.topology.left( e ) )
            continue;
        auto eAngle = angle( e );
        if ( eAngle <= bdAngle )
            report( e );
        else
        {
            if ( bd )
                report( bd );
            bd = e;
            bdAngle = eAngle;
        }
    }
}

FaceBitSet findHoleComplicatingFaces( const Mesh & mesh )
{
    MR_TIMER;

    tbb::enumerable_thread_specific<std::vector<FaceId>> threadData;
    BitSetParallelFor( findRepeatedVertsOnHoleBd( mesh.topology ), [&]( VertId v )
    {
        findHoleComplicatingFaces( mesh, v, threadData.local() );
    } );

    FaceId maxFace;
    for ( const auto & fs : threadData )
        for ( FaceId f : fs )
            maxFace = std::max( maxFace, f );

    FaceBitSet res;
    res.resize( maxFace + 1 );
    for ( const auto & fs : threadData )
        for ( FaceId f : fs )
            res.set( f );
    return res;
}

void fixMeshCreases( Mesh& mesh, const FixCreasesParams& params )
{
    auto planarAngleCos = std::cos( std::abs( PI_F - params.creaseAngle ) );
    FaceBitSet fixFacesBuffer( mesh.topology.getValidFaces().size() );
    for ( int iter = 0; iter < params.maxIters; ++iter )
    {
        auto creases = mesh.findCreaseEdges( params.creaseAngle );
        if ( creases.none() )
            return;

        for ( auto ue : creases )
        {
            if ( mesh.topology.isLoneEdge( EdgeId( ue ) ) )
                continue;
            auto findBadFaces = [&] ( EdgeId ce, bool left )
            {
                for ( auto e = ce;; )
                {
                    auto f = left ? mesh.topology.left( e ) : mesh.topology.right( e );
                    if ( !f )
                        return;
                    fixFacesBuffer.autoResizeSet( f ); // as far as we triangulate holes - new faces might appear, so we need to resize
                    e = left ? mesh.topology.next( e ) : mesh.topology.prev( e );
                    if ( e == ce )
                        return; // full cycle
                    auto nextF = left ? mesh.topology.left( e ) : mesh.topology.right( e );
                    if ( !nextF )
                        return;
                    if ( creases.test( e.undirected() ) )
                        continue;
                    if ( mesh.triangleAspectRatio( f ) > params.criticalTriAspectRatio || mesh.triangleAspectRatio( nextF ) > params.criticalTriAspectRatio )
                        continue;
                    auto digAngCos = mesh.dihedralAngleCos( e.undirected() );
                    if ( digAngCos < planarAngleCos )
                        return; // stop propagation on sharp angle
                }
            };

            int numIncidentLCreases = 0;
            int numIncidentRCreases = 0;
            auto creaseEdge = EdgeId( ue );
            for ( auto e : orgRing( mesh.topology, creaseEdge ) )
            {
                if ( creases.test( e.undirected() ) )
                    numIncidentLCreases++;
            }
            for ( auto e : orgRing( mesh.topology, creaseEdge.sym() ) )
            {
                if ( creases.test( e.undirected() ) )
                    numIncidentRCreases++;
            }
            if ( ( numIncidentRCreases > numIncidentLCreases )
                || ( numIncidentRCreases == numIncidentLCreases &&
                    mesh.topology.getOrgDegree( creaseEdge ) < mesh.topology.getOrgDegree( creaseEdge.sym() ) ) )
            {
                creaseEdge = creaseEdge.sym();// important part to triangulate worse end of the edge (mb we should change degree check to area check?)
            }
            fixFacesBuffer.reset();
            findBadFaces( creaseEdge, true );
            findBadFaces( creaseEdge, false );
            if ( fixFacesBuffer.none() )
                continue;

            auto loops = delRegionKeepBd( mesh, fixFacesBuffer, true );
            for ( const auto& loop : loops )
            {
                int i = 0;
                while ( i < loop.size() && mesh.topology.left( loop[i] ) ) ++i;
                if ( i == loop.size() )
                    continue;
                fillHole( mesh, loop[i], { .metric = getMinAreaMetric( mesh ) } );
            }
        }
    }
}

} //namespace MR
