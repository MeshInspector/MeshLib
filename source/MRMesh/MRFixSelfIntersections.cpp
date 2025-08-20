#include "MRFixSelfIntersections.h"
#include "MRMesh.h"
#include "MRMeshCollide.h"
#include "MRExpandShrink.h"
#include "MRRegionBoundary.h"
#include "MRMeshFillHole.h"
#include "MRMeshFixer.h"
#include "MRMeshRelax.h"
#include "MRMeshComponents.h"
#include "MRMeshSubdivide.h"
#include "MRTimer.h"
#include "MRBox.h"
#include "MRMapOrHashMap.h"
#include "MRPch/MRSpdlog.h"
#include <algorithm>

namespace MR
{
namespace SelfIntersections
{

Expected<FaceBitSet> getFaces( const Mesh& mesh, bool touchIsIntersection, ProgressCallback cb )
{
    auto [faceToRegionMap, componentsCount] = MeshComponents::getAllComponentsMap(
        { mesh }, MeshComponents::FaceIncidence::PerEdge );
    return findSelfCollidingTrianglesBS( mesh, cb, &faceToRegionMap, touchIsIntersection );
}

// Helper function to fix self-intersections on part of mesh
static Expected<void> doFix( Mesh& mesh, FaceBitSet& part, const Settings& settings,
    int expandSize,
    FaceBitSet& accumBadFaces,
    ProgressCallback cb );

// TODO: finish implementation
Expected<void> fixOld( Mesh& mesh, const Settings& settings )
{
    Settings currentSettings = settings;
    if ( currentSettings.subdivideEdgeLen <= 0.0f )
        currentSettings.subdivideEdgeLen = mesh.averageEdgeLength();

    // Examine current state
    float progress = 0.1f, endProgress = 0.9f;
    auto faceToRegionMap = MeshComponents::getAllComponentsMap( { mesh }, MeshComponents::FaceIncidence::PerEdge ).first;
    auto res = findSelfCollidingTrianglesBS( mesh,
        subprogress( settings.callback, 0.0f, progress ),
        &faceToRegionMap, currentSettings.touchIsIntersection );
    if ( !res.has_value() )
        return unexpected( res.error() );
    FaceBitSet badFaces = std::move( res.value() );

    // Split mesh into separate component and possibly separate areas
    FaceBitSet maxFaces = badFaces;
    expand( mesh.topology, maxFaces, currentSettings.maxExpand );
    std::vector<FaceBitSet> regions = MeshComponents::getAllComponents( { mesh, &maxFaces } );
    spdlog::info( "SelfIntersections::fix: working on {} regions", int( regions.size() ) );

    // Handle areas
    badFaces.reset();
    float progressStep = ( endProgress - progress ) / regions.size();
    for ( FaceBitSet &region : regions ) // Regions are updated in process
    {
        float progressSubStep = progressStep / currentSettings.maxExpand;
        for ( int expandSize = 1; expandSize <= currentSettings.maxExpand; expandSize++ )
        {
            region &= mesh.topology.getValidFaces(); // Ensure bit set is valid, just in case
            auto fixRes = doFix( mesh, region, currentSettings,
                         expandSize,
                         badFaces,
                         subprogress( settings.callback, progress, progress + progressSubStep ) );
            if ( true )
                return {};
            if ( !fixRes )
                return fixRes;
            progress += progressSubStep;
        }
    }

    // Handle global self-intersections (bridge-like between areas)
    if ( settings.method == Settings::Method::Relax )
        return {};
    faceToRegionMap = MeshComponents::getAllComponentsMap( { mesh }, MeshComponents::FaceIncidence::PerEdge ).first;
    res = findSelfCollidingTrianglesBS( mesh,
        subprogress( settings.callback, endProgress, 1.0f ),
        &faceToRegionMap, settings.touchIsIntersection );
    if ( !res.has_value() )
        return unexpected( res.error() );
    badFaces = res.value() - badFaces; // Exclude incurable faces
    expand( mesh.topology, badFaces, 1 );
    // Just clear regions and fill holes
    std::vector<EdgeLoop> boundary = findRightBoundary( mesh.topology, &badFaces );
    mesh.topology.deleteFaces( badFaces );
    for ( const EdgeLoop& loop : boundary )
    {
        if ( std::all_of( loop.begin(), loop.end(),
            [&] ( EdgeId e ) { return mesh.topology.isLeftInRegion( e ); } ) )
        {
            if ( mesh.topology.left( loop.front().sym() ) )
                // Invalid loop, ignoring
                continue;
            fillHole( mesh, loop.front().sym(), { .metric = getMinAreaMetric( mesh ) } );
        }
    }

    return {};
}

Expected<void> fix( Mesh& mesh, const Settings& settings )
{
    MR_TIMER;

    if ( !reportProgress( settings.callback, 0.0f ) )
        return unexpectedOperationCanceled();

    if ( settings.touchIsIntersection )
    {
        FixMeshDegeneraciesParams fdParams;
        fdParams.maxDeviation = mesh.getBoundingBox().diagonal() * 1e-4f;
        fdParams.tinyEdgeLength = fdParams.maxDeviation * 0.1f;

        // do not subdivide if it is explicitly forbidden by user
        fdParams.mode = FixMeshDegeneraciesParams::Mode::Decimate;
        if ( settings.subdivideEdgeLen < FLT_MAX )
            fdParams.mode = FixMeshDegeneraciesParams::Mode::Remesh;

        fdParams.cb = subprogress( settings.callback, 0.0f, 0.2f );
        auto fdRes = fixMeshDegeneracies( mesh, fdParams );
        if ( !fdRes.has_value() )
            return fdRes;
    }

    auto faceToRegionMap = MeshComponents::getAllComponentsMap( { mesh } ).first;

    if ( !reportProgress( settings.callback, 0.25f ) )
        return unexpectedOperationCanceled();

    auto res = findSelfCollidingTrianglesBS( mesh,
                                             subprogress( settings.callback, 0.25f, 0.4f ),
                                             &faceToRegionMap, settings.touchIsIntersection );
    if ( !res.has_value() )
        return unexpected( res.error() );

    if ( res->none() )
        return {};

    expand( mesh.topology, *res, settings.maxExpand );

    Settings currentSettings = settings;
    if ( currentSettings.subdivideEdgeLen < FLT_MAX )
    {
        auto box = mesh.computeBoundingBox( &res.value() );
        if ( currentSettings.subdivideEdgeLen <= 0.0f )
            currentSettings.subdivideEdgeLen = box.valid() ? box.diagonal() * 1e-2f : mesh.getBoundingBox().diagonal() * 1e-4f;

        SubdivideSettings ssettings;
        ssettings.region = &res.value();
        ssettings.maxEdgeLen = currentSettings.subdivideEdgeLen;
        ssettings.maxEdgeSplits = 1000;
        ssettings.maxDeviationAfterFlip = ssettings.maxEdgeLen;
        ssettings.criticalAspectRatioFlip = FLT_MAX;
        ssettings.progressCallback = subprogress( settings.callback, 0.4f, 0.5f );
        subdivideMesh( mesh, ssettings );
    }

    if ( !reportProgress( settings.callback, 0.5f ) )
        return unexpectedOperationCanceled();

    faceToRegionMap = MeshComponents::getAllComponentsMap( { mesh } ).first;

    if ( !reportProgress( settings.callback, 0.55f ) )
        return unexpectedOperationCanceled();

    res = findSelfCollidingTrianglesBS( MeshPart( mesh, &res.value() ),
                                        subprogress( settings.callback, 0.55f, 0.7f ),
                                        &faceToRegionMap,settings.touchIsIntersection );

    if ( !res.has_value() )
        return unexpected( res.error() );

    expand( mesh.topology, *res, settings.maxExpand );

    if ( settings.method == Settings::Method::Relax )
    {
        auto verts = getIncidentVerts( mesh.topology, *res );

        if ( !reportProgress( settings.callback, 0.8f ) )
            return unexpectedOperationCanceled();
        MeshRelaxParams params;
        params.iterations = settings.relaxIterations;
        params.region = &verts;
        if ( !relax( mesh, params, subprogress( settings.callback, 0.8f, 1.0f ) ) )
            return unexpectedOperationCanceled();
    }
    else
    {
        auto boundaryEdges = mesh.topology.findLeftBdEdges();
        mesh.topology.deleteFaces( *res );
        mesh.topology.deleteFaces( findHoleComplicatingFaces( mesh ) );
        mesh.invalidateCaches();
        auto holes = findRightBoundary( mesh.topology );

        if ( !reportProgress( settings.callback, 0.8f ) )
            return unexpectedOperationCanceled();

        auto sp = subprogress( settings.callback, 0.8f, 0.95f );
        for ( int i = 0; i < holes.size(); ++i )
        {
            bool outerBounds = false;
            for ( auto e : holes[i] )
            {
                if ( boundaryEdges.test( e ) )
                {
                    outerBounds = true;
                    break;
                }
            }
            if ( outerBounds )
                continue; // part of mesh boundary

            // Fill hole
            // MultipleEdgesResolveMode::Simple should be enough after deleting findHoleComplicatingFaces(...)
            // But if multiple edges appear often, could be changed to MultipleEdgesResolveMode::Strong
            fillHole( mesh, holes[i].front(), {.metric = getMinAreaMetric(mesh),
                .multipleEdgesResolveMode = FillHoleParams::MultipleEdgesResolveMode::Simple });

            if ( !reportProgress( sp, float( i + 1 ) / float( holes.size() ) ) )
                return unexpectedOperationCanceled();
        }

        if ( !reportProgress( settings.callback, 1.0f ) )
            return unexpectedOperationCanceled();
    }
    return {};
}

// Helper function to find own self-intersections on a mesh part
static Expected<FaceBitSet> findSelfCollidingTrianglesBSForPart( Mesh& mesh, const FaceBitSet& part, ProgressCallback cb, bool touchIsIntersection )
{
    FaceMapOrHashMap tgt2srcFaces;
    PartMapping mapping;
    mapping.tgt2srcFaces = &tgt2srcFaces;
    Mesh partMesh = mesh.cloneRegion( part, false, mapping );
    // Faster than searching in mesh part due to AABB tree rebuild
    auto res = findSelfCollidingTrianglesBS( { partMesh }, cb, nullptr, touchIsIntersection );
    if ( !res.has_value() )
        return unexpected( res.error() );
    FaceBitSet result( mesh.topology.lastValidFace() + 1 );
    if ( auto map = tgt2srcFaces.getMap() )
    {
        for ( FaceId f : *res )
            result.set( (*map)[f] );
    }
    else
        assert( false );
    return result;
}

static Expected<void> doFix( Mesh &mesh, FaceBitSet &part, const Settings & settings,
    int expandSize,
    FaceBitSet& accumBadFaces,
    ProgressCallback cb )
{
    // Find colliding triangles
    float progress = 0.1f, endProgress = 0.9f;
    auto res = findSelfCollidingTrianglesBSForPart( mesh, part, 
        subprogress( cb, 0.0f, progress ), settings.touchIsIntersection );
    if ( !res.has_value() )
        return unexpected( res.error() );
    // Get area around colliding triangles
    FaceBitSet facesAround = std::move( res.value() );
    if ( facesAround.empty() )
        return {};
    expand( mesh.topology, facesAround, expandSize );
    // Iterate through regions around colliding triangles
    std::vector<FaceBitSet> regions = MeshComponents::getAllComponents( { mesh, &facesAround } );
    float progressStep = ( endProgress - progress ) / regions.size();
    for ( const FaceBitSet &region : regions )
    {
        if ( !reportProgress( cb, progress ) )
            return unexpectedOperationCanceled();
        progress += progressStep;

        std::vector<EdgeLoop> boundary = findLeftBoundary( mesh.topology, &region );
        if ( boundary.empty() )
        {
            // Small separate component
            mesh.topology.deleteFaces( region );
            part &= mesh.topology.getValidFaces();
            mesh.invalidateCaches();
            continue;
        }
        bool isRegionOnBorder = std::any_of( boundary.begin(), boundary.end(),
            [&] ( EdgeLoop& loop )
            {
                // Check if loop touches mesh border (but exclude `region` inner holes)
                size_t countOnBorder = std::count_if( loop.begin(), loop.end(),
                        [&] ( EdgeId e ) { return !mesh.topology.isLeftInRegion( e.sym() ); } );
                return countOnBorder != 0 && countOnBorder != loop.size();
            } );
        if ( isRegionOnBorder || settings.method == Settings::Method::Relax )
        {
            // Region is on border, cannot use fillHole
            VertBitSet verts = getIncidentVerts( mesh.topology, region );
            // Do not touch mesh border
            for ( EdgeLoop &loop: boundary )
                for ( EdgeId e : loop )
                    if ( !mesh.topology.isLeftInRegion( e.sym() ) )
                    {
                        verts.reset( mesh.topology.org( e ) );
                        verts.reset( mesh.topology.dest( e ) );
                    }
            // Try fixing by relax
            relax( mesh, { { settings.relaxIterations, &verts } } );
        }
        else
        {
            // Inner region - fill hole
            mesh.topology.deleteFaces( region );
            part &= mesh.topology.getValidFaces();
            mesh.invalidateCaches();

            for ( EdgeLoop& loop : boundary )
            {
                if ( mesh.topology.left( loop.front().sym() ) )
                    // Invalid loop, ignoring
                    continue;
                FaceBitSet newFaces;
                fillHole( mesh, loop.front(), {
                    .metric = getMinAreaMetric( mesh ),
                    .outNewFaces = &newFaces } );

                VertBitSet newVerts;
                SubdivideSettings subdivideSettings;
                subdivideSettings.maxEdgeLen = settings.subdivideEdgeLen;
                subdivideSettings.maxEdgeSplits = 1000000;
                subdivideSettings.region = &newFaces;
                subdivideSettings.newVerts = &newVerts;
                subdivideMesh( mesh, subdivideSettings );
                part |= newFaces;

                relax( mesh, { { 3, &newVerts } } );
            }
        }
        mesh.invalidateCaches();
    }
    // Find resulting self-intersections
    res = findSelfCollidingTrianglesBSForPart( mesh, part,
        subprogress( cb, endProgress, 1.0 ), settings.touchIsIntersection );
    if ( !res.has_value() )
        return unexpected( res.error() );
    accumBadFaces |= res.value();
    return {};
}

}
}
