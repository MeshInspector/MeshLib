#include "MRRebuildMesh.h"
#include "MROffset.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRMapEdge.h"
#include "MRMesh/MRMeshDecimate.h"
#include "MRMesh/MRMeshCollide.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMeshSubdivide.h"
#include "MRMesh/MRMeshTotalAngle.h"

namespace MR
{

Expected<Mesh> rebuildMesh( const MeshPart& mp, const RebuildMeshSettings& settings )
{
    MR_TIMER;

    auto progress = settings.progress;

    GeneralOffsetParameters genOffsetParams;
    switch ( settings.signMode )
    {
    default:
        assert( false );
        [[fallthrough]];
    case SignDetectionModeShort::Auto:
        if ( mp.mesh.topology.isClosed( mp.region ) )
        {
            auto expSelfy = findSelfCollidingTriangles( mp, nullptr, subprogress( progress, 0.0f, 0.1f ) );
            progress = subprogress( progress, 0.1f, 1.0f );
            if ( !expSelfy )
                return unexpected( std::move( expSelfy.error() ) );
            if ( *expSelfy )
                genOffsetParams.signDetectionMode = SignDetectionMode::HoleWindingRule;
            else if ( settings.offsetMode == OffsetMode::Smooth )
                genOffsetParams.signDetectionMode = SignDetectionMode::OpenVDB;
            else
                genOffsetParams.signDetectionMode = SignDetectionMode::ProjectionNormal;
        }
        else
            genOffsetParams.signDetectionMode = SignDetectionMode::HoleWindingRule;
        break;

    case SignDetectionModeShort::HoleWindingNumber:
        genOffsetParams.signDetectionMode = SignDetectionMode::HoleWindingRule;
        break;

    case SignDetectionModeShort::ProjectionNormal:
        genOffsetParams.signDetectionMode = SignDetectionMode::ProjectionNormal;
        break;
    }
    if ( settings.onSignDetectionModeSelected )
        settings.onSignDetectionModeSelected( genOffsetParams.signDetectionMode );

    std::optional<Mesh> subMesh;
    if ( settings.preSubdivide && genOffsetParams.signDetectionMode != SignDetectionMode::OpenVDB ) // OpenVDB slows down with more input triangles
    {
        if ( auto maybeMesh = copySubdividePackMesh( mp, settings.voxelSize, subprogress( progress, 0.0f, 0.1f ) ) )
            subMesh = std::move( *maybeMesh );
        else
            return unexpected( std::move( maybeMesh.error() ) );
        progress = subprogress( progress, 0.1f, 1.0f );
    }

    const float postprocessDuration =
        ( settings.decimate ? 0.3f : 0.0f ) +
        ( settings.reduceAngleNumIters > 0 ? 0.1f : 0.0f );
    const auto offsetProgress = subprogress( progress, 0.0f, 1 - postprocessDuration );
    const auto postprocessProgress = subprogress( progress, 1 - postprocessDuration, 1.0f );
        
    genOffsetParams.closeHolesInHoleWindingNumber = settings.closeHolesInHoleWindingNumber;
    genOffsetParams.voxelSize = settings.voxelSize;
    genOffsetParams.mode = settings.offsetMode;
    genOffsetParams.windingNumberThreshold = settings.windingNumberThreshold;
    genOffsetParams.windingNumberBeta = settings.windingNumberBeta;
    genOffsetParams.fwn = settings.fwn;
    genOffsetParams.callBack = offsetProgress;

    UndirectedEdgeBitSet sharpEdges;
    genOffsetParams.outSharpEdges = &sharpEdges;

    auto resMesh = generalOffsetMesh( subMesh ? *subMesh : mp, 0.0f, genOffsetParams );
    if ( !resMesh.has_value() )
        return resMesh;

    auto maybeOk = postprocessMeshFromVoxels( *resMesh, MeshFromVoxelsPostProcessingParams
        {
            .voxelSize = genOffsetParams.voxelSize,
            .reduceAngleNumIters = settings.reduceAngleNumIters,
            .decimate = settings.decimate,
            .tinyEdgeLength = settings.tinyEdgeLength,
            .sharpEdges = sharpEdges.any() ? &sharpEdges : nullptr
        }, postprocessProgress );
    if ( !maybeOk )
        return unexpected( std::move( maybeOk.error() ) );

    if ( settings.outSharpEdges )
        *settings.outSharpEdges = std::move( sharpEdges );

    return resMesh;
}

Expected<void> postprocessMeshFromVoxels( Mesh& mesh, const MeshFromVoxelsPostProcessingParams& params, const ProgressCallback& progress )
{
    MR_TIMER;
    assert( params.voxelSize > 0 );
    if ( params.voxelSize <= 0 )
        return unexpected( "voxelize is not set" );

    ProgressCallback decimateProgress, postDecimateReduceAngleProgress;
    const auto notFlippable = params.sharpEdges && params.sharpEdges->any() ? params.sharpEdges : nullptr;
    ReduceTotalAngleParams reduceTotalAngleParams
    {
        .notFlippable = notFlippable
    };
    if ( params.reduceAngleNumIters > 0 )
    {
        reduceTotalAngleInMesh( mesh, params.reduceAngleNumIters, reduceTotalAngleParams, subprogress( progress, 0.0f, 0.1f ) );
        if ( !reportProgress( progress, 0.1f ) )
            return unexpectedOperationCanceled();
        decimateProgress = subprogress( progress, 0.1f, 0.95f );
        postDecimateReduceAngleProgress = subprogress( progress, 0.95f, 1.0f );
    }
    else
        decimateProgress = progress;

    if ( params.decimate && mesh.topology.numValidFaces() > 0 )
    {
        const auto map = mesh.packOptimally( false );
        if ( !reportProgress( decimateProgress, 0.1f ) )
            return unexpectedOperationCanceled();

        if ( params.sharpEdges )
            *params.sharpEdges = mapEdges( map.e, *params.sharpEdges );

        DecimateSettings decimSettings
        {
            .maxError = 0.25f * params.voxelSize,
            .tinyEdgeLength = params.tinyEdgeLength,
            .stabilizer = 1e-5f, // 1e-6 here resulted in a bit worse mesh
            .notFlippable = notFlippable,
            .packMesh = true,
            .progressCallback = subprogress( decimateProgress, 0.1f, 1.0f ),
            .subdivideParts = 64
        };
        if ( decimateMesh( mesh, decimSettings ).cancelled )
            return unexpectedOperationCanceled();

        if ( params.reduceAngleNumIters > 0 )
        {
            reduceTotalAngleInMesh( mesh, params.reduceAngleNumIters, reduceTotalAngleParams, postDecimateReduceAngleProgress );
            if ( !reportProgress( postDecimateReduceAngleProgress, 1.0f ) )
                return unexpectedOperationCanceled();
        }
    }

    return {};
}

} //namespace MR
