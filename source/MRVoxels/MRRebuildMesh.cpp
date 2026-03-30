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

    ProgressCallback preDecimateProgress, decimateProgress;
    if ( settings.decimate )
    {
        preDecimateProgress = subprogress( progress, 0.0f, 0.7f );
        decimateProgress = subprogress( progress, 0.7f, 1.0f );
    }
    else
        preDecimateProgress = progress;

    ProgressCallback preDecimateReduceAngleProgress, postDecimateReduceAngleProgress;
    if ( settings.reduceAngleNumIters > 0 )
    {
        preDecimateReduceAngleProgress = subprogress( preDecimateProgress, 0.9f, 1.0f );
        preDecimateProgress = subprogress( preDecimateProgress, 0.0f, 0.9f );
        postDecimateReduceAngleProgress = subprogress( decimateProgress, 0.9f, 1.0f );
        decimateProgress = subprogress( decimateProgress, 0.0f, 0.9f );
    }

    genOffsetParams.closeHolesInHoleWindingNumber = settings.closeHolesInHoleWindingNumber;
    genOffsetParams.voxelSize = settings.voxelSize;
    genOffsetParams.mode = settings.offsetMode;
    genOffsetParams.windingNumberThreshold = settings.windingNumberThreshold;
    genOffsetParams.windingNumberBeta = settings.windingNumberBeta;
    genOffsetParams.fwn = settings.fwn;
    genOffsetParams.callBack = preDecimateProgress;

    UndirectedEdgeBitSet sharpEdges;
    genOffsetParams.outSharpEdges = &sharpEdges;

    auto resMesh = generalOffsetMesh( subMesh ? *subMesh : mp, 0.0f, genOffsetParams );
    if ( !resMesh.has_value() )
        return resMesh;

    ReduceTotalAngleParams reduceTotalAngleParams
    {
        .notFlippable = sharpEdges.any() ? &sharpEdges : nullptr
    };
    if ( settings.reduceAngleNumIters > 0 )
    {
        reduceTotalAngleInMesh( *resMesh, settings.reduceAngleNumIters, reduceTotalAngleParams, preDecimateReduceAngleProgress );
        if ( !reportProgress( preDecimateReduceAngleProgress, 1.0f ) )
            return unexpectedOperationCanceled();
    }

    if ( settings.decimate && resMesh->topology.numValidFaces() > 0 )
    {
        const auto map = resMesh->packOptimally( false );
        if ( !reportProgress( decimateProgress, 0.1f ) )
            return unexpectedOperationCanceled();

        sharpEdges = mapEdges( map.e, sharpEdges );

        DecimateSettings decimSettings
        {
            .maxError = 0.25f * genOffsetParams.voxelSize,
            .tinyEdgeLength = settings.tinyEdgeLength,
            .stabilizer = 1e-5f, // 1e-6 here resulted in a bit worse mesh
            .notFlippable = sharpEdges.any() ? &sharpEdges : nullptr,
            .packMesh = true,
            .progressCallback = subprogress( decimateProgress, 0.1f, 1.0f ),
            .subdivideParts = 64
        };
        if ( decimateMesh( *resMesh, decimSettings ).cancelled )
            return unexpectedOperationCanceled();

        if ( settings.reduceAngleNumIters > 0 )
        {
            reduceTotalAngleInMesh( *resMesh, settings.reduceAngleNumIters, reduceTotalAngleParams, postDecimateReduceAngleProgress );
            if ( !reportProgress( postDecimateReduceAngleProgress, 1.0f ) )
                return unexpectedOperationCanceled();
        }
    }

    if ( settings.outSharpEdges )
        *settings.outSharpEdges = std::move( sharpEdges );

    return resMesh;
}

} //namespace MR
