#include "MRMeshFixer.h"
#include "MRVertDuplication.h"
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
#include "MRMeshPatch.h"
#include "MRObjectMeshData.h"
#include "MRMeshSubdivideCallbacks.h"
#include "MRProjectionMeshAttribute.h"
#include "MRPartMapping.h"
#include "MRMapOrHashMap.h"

namespace MR
{

// returns the first edge with the origin in given vertex and without left face
// if the number of such edges in the vertex ring is larger than given limit, otherwise returns invalid edge
static EdgeId findNoLeftEdgeAboveLimit( const MeshTopology & m, VertId a, int limit )
{
    EdgeId e0 = m.edgeWithOrg( a );
    if ( !e0.valid() )
        return {}; //invalid vertex

    EdgeId eh; // first found edge without left face
    int holes = 0;
    EdgeId e = e0;
    for (;;)
    {
        if ( !m.left( e ).valid() )
        {
            if ( !eh.valid() )
                eh = e;
            if ( ++holes > limit )
                return eh;
        }
        e = m.next( e );
        if ( e == e0 )
            return {};
    }
}

int duplicateMultiHoleVertices( Mesh & mesh, int maxHoles, std::vector<MeshBuilder::VertDuplication> * dups )
{
    MR_TIMER;
    assert( maxHoles >= 1 );
    if ( dups )
        dups->clear();

    VertBitSet vertsForDup( mesh.topology.vertSize() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        if ( findNoLeftEdgeAboveLimit( mesh.topology, v, maxHoles ).valid() )
            vertsForDup.set( v );
    } );

    int duplicates = 0;
    for ( auto v : vertsForDup )
    {
        for (;;)
        {
            EdgeId e1 = findNoLeftEdgeAboveLimit( mesh.topology, v, maxHoles );
            if ( !e1.valid() )
                break;

            EdgeId e0 = e1;
            while ( mesh.topology.right( e0 ).valid() )
                e0 = mesh.topology.prev( e0 );

            // unsplice [e0, e1] and create new vertex for it
            mesh.topology.splice( mesh.topology.prev( e0 ), e1 );
            assert( !mesh.topology.org( e0 ).valid() );

            auto vDup = mesh.addPoint( mesh.points[v] );
            mesh.topology.setOrg( e0, vDup );
            if ( dups )
                dups->push_back( { .srcVert = v, .dupVert = vDup } );

            ++duplicates;
        }
    }

    return duplicates;
}

int duplicateMultiHoleVertices( Mesh & mesh )
{
    return duplicateMultiHoleVertices( mesh, 1 );
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

// returns the longest edge of given triangle, having the triangle on its left
static EdgeId longestEdge( const Mesh& mesh, FaceId f )
{
    EdgeId res;
    float maxLengthSq = -1;
    for ( auto e : leftRing( mesh.topology, f ) )
    {
        if ( const auto lengthSq = mesh.edgeLengthSq( e ); lengthSq > maxLengthSq )
        {
            maxLengthSq = lengthSq;
            res = e;
        }
    }
    return res;
}

// decimation can fix a degenerate triangle with its vertex near the middle of the opposite (longest) edge only by flipping that edge;
// so if the edge is not flippable, splits it at the projection of the vertex, and the new short edge can be collapsed instead;
// onEdgeSplit( e1, e, t ) is called after each split of (e) into (e1->e) at org(e1) + t * ( dest(e) - org(e1) )
static void splitNotFlippableLongEdges( Mesh& mesh, FaceBitSet& region, UndirectedEdgeBitSet& notFlippable,
    float criticalAspectRatio, float shortEdgeLength, const std::function<void( EdgeId e1, EdgeId e, float t )>& onEdgeSplit )
{
    MR_TIMER;
    MR_WRITER( mesh );
    const auto shortEdgeLengthSq = sqr( double( shortEdgeLength ) );
    const auto degenerateFaces = findDegenerateFaces( { mesh, &region }, criticalAspectRatio ).value();
    for ( auto f : degenerateFaces )
    {
        const auto e = longestEdge( mesh, f );
        if ( !notFlippable.test( e.undirected() ) || mesh.edgeLengthSq( e ) <= shortEdgeLengthSq )
            continue; // the longest edge can be flipped, or all edges are short (or even coincide)
        const Vector3d a( mesh.orgPnt( e ) ), b( mesh.destPnt( e ) ), p( mesh.destPnt( mesh.topology.next( e ) ) );
        const auto t = dot( p - a, b - a ) / ( b - a ).lengthSq();
        const auto q = a + t * ( b - a );
        if ( ( p - q ).lengthSq() > shortEdgeLengthSq || ( q - a ).lengthSq() <= shortEdgeLengthSq || ( q - b ).lengthSq() <= shortEdgeLengthSq )
            continue; // the new edge would not be short enough to collapse, or the triangle already has a short edge
        const auto e1 = mesh.splitEdge( e, Vector3f( q ), &region );
        notFlippable.autoResizeSet( e1.undirected() );
        if ( onEdgeSplit )
            onEdgeSplit( e1, e, float( t ) );
    }
}

// if data is given, then its attributes are kept valid, and it must own the mesh
static Expected<void> fixDegeneracies( Mesh& mesh, ObjectMeshData* data, const FixMeshDegeneraciesParams& params )
{
    assert( !data || data->mesh.get() == &mesh );
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

    MeshAttributesToUpdate attributes;
    // updates the attributes of data after the split of (e) into (e1->e)
    OnEdgeSplit onEdgeSplit;
    // the edges of data that must not be flipped with params.protectAttributeBorders
    UndirectedEdgeBitSet notFlippableEdges;
    UndirectedEdgeBitSet* notFlippable = nullptr;
    if ( data )
    {
        resizeAttributesToMesh( *data );
        if ( !data->uvCoordinates.empty() )
            attributes.uvCoords = &data->uvCoordinates;
        if ( !data->vertColors.empty() )
            attributes.colorMap = &data->vertColors;
        if ( !data->texturePerFace.empty() )
            attributes.texturePerFace = &data->texturePerFace;
        if ( !data->faceColors.empty() )
            attributes.faceColors = &data->faceColors;
        onEdgeSplit = [&, updateAttributes = meshOnEdgeSplitAttribute( mesh, attributes )] ( EdgeId e1, EdgeId e )
        {
            updateAttributes( e1, e );
            if ( contains( data->selectedFaces, mesh.topology.left( e ) ) )
                data->selectedFaces.autoResizeSet( mesh.topology.left( e1 ) );
            if ( contains( data->selectedFaces, mesh.topology.right( e ) ) )
                data->selectedFaces.autoResizeSet( mesh.topology.right( e1 ) );
            for ( auto s : { &data->selectedEdges, &data->creases } )
                if ( s->test( e.undirected() ) )
                    s->autoResizeSet( e1.undirected() );
        };

        if ( params.protectAttributeBorders )
        {
            // a flip keeps the ids of the edge and of its faces, so a flipped edge between different colors or textures
            // would extend the attribute of one face over a part of the other one, and a flipped selected edge or crease would cross them
            notFlippableEdges = data->selectedEdges | data->creases
                | edgesBetweenDifferentColors( mesh.topology, data->faceColors )
                | edgesBetweenDifferentTextures( mesh.topology, data->texturePerFace );
            if ( notFlippableEdges.any() )
                notFlippable = &notFlippableEdges;
        }
    }

    if ( notFlippable )
    {
        splitNotFlippableLongEdges( mesh, *regRes, *notFlippable, params.criticalTriAspectRatio,
            std::max( params.maxDeviation, params.tinyEdgeLength ), [&] ( EdgeId e1, EdgeId e, float t )
        {
            onEdgeSplit( e1, e );
            // the new vertex is not in the middle of the edge
            const auto a = mesh.topology.org( e1 );
            const auto b = mesh.topology.dest( e );
            if ( auto uv = attributes.uvCoords )
                uv->back() = ( *uv )[a] * ( 1 - t ) + ( *uv )[b] * t;
            if ( auto colors = attributes.colorMap )
                colors->back() = ( *colors )[a] * ( 1 - t ) + ( *colors )[b] * t;
        } );
    }

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
    if ( notFlippable )
    {
        dsettings.notFlippable = notFlippable;
        dsettings.collapseNearNotFlippable = true; // otherwise the short edges ending on not flippable edges would remain
    }
    if ( data )
    {
        dsettings.onEdgeDel = [&] ( EdgeId del, EdgeId rem )
        {
            for ( auto s : { &data->selectedEdges, &data->creases } )
                if ( s->test_set( del.undirected(), false ) && rem )
                    s->autoResizeSet( rem.undirected() );
        };
    }

    auto res = data ? decimateObjectMeshData( *data, dsettings ) : decimateMesh( mesh, dsettings );

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
        .onEdgeSplit = onEdgeSplit,
        .progressCallback = subprogress( sbs, 0.25f, 1.0f )
    };
    // subdivision cannot fix degenerate triangles with not flippable longest edges, and they must not keep it splitting other edges
    FaceBitSet subdivisionRegion;
    if ( notFlippable )
    {
        ssettings.notFlippable = notFlippable;
        subdivisionRegion = mesh.topology.getFaceIds( params.region );
        BitSetParallelFor( findDegenerateFaces( { mesh, params.region }, params.criticalTriAspectRatio ).value(), [&] ( FaceId f )
        {
            if ( notFlippable->test( longestEdge( mesh, f ).undirected() ) )
                subdivisionRegion.reset( f );
        } );
        ssettings.region = &subdivisionRegion;
        ssettings.maintainRegion = params.region;
    }
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

    FillHoleNicelySettings psettings
    {
        .triangulateParams =
        {
            .multipleEdgesResolveMode = FillHoleParams::MultipleEdgesResolveMode::Strong,
        },
        .subdivideSettings =
        {
            .notFlippable = notFlippable,
            .maxEdgeLen = 0.0f, // to use default from `patchMesh`
            .maxEdgeSplits = 20'000,
            .onEdgeSplit = onEdgeSplit, // patch subdivision can split the faces around the patch too
        }
    };

    // the removed surface, in data mode the new elements get the attributes of its nearest points
    const bool projectAttributes = data && ( attributes.uvCoords || attributes.colorMap
        || attributes.texturePerFace || attributes.faceColors || data->selectedFaces.any() );
    Mesh patchRefMesh;
    FaceMapOrHashMap refFaces;
    VertMapOrHashMap refVerts;
    if ( params.mimicPatch || projectAttributes )
    {
        PartMapping map;
        if ( projectAttributes )
        {
            map.tgt2srcFaces = &refFaces;
            map.tgt2srcVerts = &refVerts;
        }
        patchRefMesh.addMeshPart( { mesh,&*regRes }, map );
    }
    if ( params.mimicPatch )
    {
        psettings.triangulateParams.metric = mixMetrics(
                getCircumscribedMetric( mesh ), getCloseSurfaceFillMetric( mesh, patchRefMesh ),
                [] ( double a, double b )->double
                {
                    return a + 100.0 * std::sqrt( b );
                } );
        psettings.smoothCurvature = false;
    }
    else
    {
        psettings.triangulateParams.metric = getUniversalMetric( mesh );
        psettings.smoothCurvature = true;
        psettings.smoothSettings.edgeWeights = EdgeWeights::Unit; // use unit weights to avoid potential laplacian degeneration (which leads to nan coords)
    }

    const bool updateSelection = data && data->selectedFaces.any();
    const auto removedSelection = updateSelection ? data->selectedFaces & *regRes : FaceBitSet{};
    const auto vertSize0 = mesh.topology.vertSize();

    auto newFaces = patchMesh( mesh, *regRes, psettings );

    if ( projectAttributes )
    {
        resizeAttributesToMesh( *data );
        if ( updateSelection )
            data->selectedFaces.resize( mesh.topology.faceSize() );

        // the attributes of removed elements are still stored in data
        const auto & refToFace = *refFaces.getMap();
        projectFaceAttribute( MeshPart( mesh, &newFaces ), patchRefMesh, [&] ( FaceId f, const MeshProjectionResult & res )
        {
            const auto src = refToFace[res.proj.face];
            if ( attributes.texturePerFace )
                ( *attributes.texturePerFace )[f] = ( *attributes.texturePerFace )[src];
            if ( attributes.faceColors )
                ( *attributes.faceColors )[f] = ( *attributes.faceColors )[src];
            if ( updateSelection )
                data->selectedFaces.set( f, removedSelection.test( src ) );
        } );

        const auto vertSize = mesh.topology.vertSize();
        if ( ( attributes.uvCoords || attributes.colorMap ) && vertSize > vertSize0 )
        {
            VertBitSet newVerts( vertSize );
            newVerts.set( VertId( vertSize0 ), vertSize - vertSize0, true );
            newVerts &= mesh.topology.getValidVerts();
            const auto & refToVert = *refVerts.getMap();
            projectVertAttribute( MeshVertPart( mesh, &newVerts ), patchRefMesh,
                [&] ( VertId v, const MeshProjectionResult & res, VertId v0, VertId v1, VertId v2 )
            {
                const auto a = refToVert[v0], b = refToVert[v1], c = refToVert[v2];
                if ( auto uv = attributes.uvCoords )
                    ( *uv )[v] = res.mtp.bary.interpolate( ( *uv )[a], ( *uv )[b], ( *uv )[c] );
                if ( auto colors = attributes.colorMap )
                    ( *colors )[v] = res.mtp.bary.interpolate( ( *colors )[a], ( *colors )[b], ( *colors )[c] );
            } );
        }
    }

    if ( data )
    {
        data->selectedFaces &= mesh.topology.getValidFaces();
        mesh.topology.excludeLoneEdges( data->selectedEdges );
        mesh.topology.excludeLoneEdges( data->creases );
    }
    if ( params.region )
    {
        *params.region |= newFaces;
        *params.region &= mesh.topology.getValidFaces();
    }
    return {};
}

Expected<void> fixMeshDegeneracies( Mesh& mesh, const FixMeshDegeneraciesParams& params )
{
    MR_TIMER;
    return fixDegeneracies( mesh, nullptr, params );
}

Expected<void> fixMeshDataDegeneracies( ObjectMeshData& data, const FixMeshDegeneraciesParams& params )
{
    MR_TIMER;
    if ( !data.mesh )
    {
        assert( false );
        return unexpected( "No mesh in ObjectMeshData" );
    }
    return fixDegeneracies( *data.mesh, &data, params );
}

VertBitSet findInnerVertsOfDegree( const MeshTopology& topology, int n, const VertBitSet* region /*= nullptr */ )
{
    const auto& zone = topology.getVertIds( region );
    VertBitSet result( zone.size() );
    BitSetParallelFor( zone, [&] ( VertId v )
    {
        if ( topology.isVertInnerAndHasDegree( v, n ) )
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
        rayMeshIntersectAllPrecise( *targetMesh, Line3d( triCenter, normal ), interPred );
        bool pValid = counter % 2 == 0;
        auto pCounter = counter;
        bool nValid = true;
        int nCounter = INT_MAX;
        bool resValid = pValid;
        if ( params.mode != FindDisorientationParams::RayMode::Positive )
        {
            counter = 0;
            rayMeshIntersectAllPrecise( *targetMesh, Line3d( triCenter, -normal ), interPred );
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

void fixMultipleEdges( Mesh & mesh, const std::vector<MultipleEdge> & multipleEdges, FaceHashMap * new2Old )
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
            mesh.splitEdge( e.sym(), nullptr, new2Old );
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

Expected<FaceBitSet> findNotSmoothFaces( const MeshPart& mp, float minAngle, ProgressCallback cb )
{
    MR_TIMER;
    FaceBitSet res( mp.mesh.topology.faceSize() );
    auto completed = BitSetParallelFor( mp.mesh.topology.getFaceIds( mp.region ), [&] ( FaceId f )
    {
        if ( !mp.mesh.topology.hasFace( f ) )
            return;
        EdgeId es[3];
        mp.mesh.topology.getTriEdges( f, es );
        Vector3f nc = mp.mesh.normal( f );
        Vector3f n[3];
        float a0 = 0;
        for ( int i = 0; i < 3; ++i )
        {
            auto r = mp.mesh.topology.right( es[i] );
            if ( !r )
                return; // f is boundary triangle
            n[i] = mp.mesh.normal( r );
            a0 += angle( nc, n[i] );
        }
        float a1 = angle( n[0], n[1] ) + angle( n[1], n[2] ) + angle( n[2], n[0] );
        if ( a0 > a1 + minAngle )
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

void deleteFacesWithLongEdges( Mesh& mesh, float maxEdgeLength )
{
    MR_TIMER;
    const auto maxEdgeLengthSq = sqr( maxEdgeLength );
    FaceBitSet longFaces( mesh.topology.faceSize() );
    BitSetParallelFor( mesh.topology.getValidFaces(), [&] ( FaceId f )
    {
        for ( EdgeId e : leftRing( mesh.topology, f ) )
        {
            if ( mesh.edgeLengthSq( e.undirected() ) > maxEdgeLengthSq )
            {
                longFaces.set( f );
                break;
            }
        }
    } );
    mesh.deleteFaces( longFaces );
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
