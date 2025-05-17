#include "MRRebuildMesh.h"
#include "MROffset.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRMapEdge.h"
#include "MRMesh/MRMeshDecimate.h"
#include "MRMesh/MRMeshCollide.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMeshSubdivide.h"

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

    genOffsetParams.closeHolesInHoleWindingNumber = settings.closeHolesInHoleWindingNumber;
    genOffsetParams.voxelSize = settings.voxelSize;
    genOffsetParams.mode = settings.offsetMode;
    genOffsetParams.windingNumberThreshold = settings.windingNumberThreshold;
    genOffsetParams.windingNumberBeta = settings.windingNumberBeta;
    genOffsetParams.fwn = settings.fwn;
    genOffsetParams.callBack = subprogress( progress, 0.0f, ( settings.decimate ? 0.7f : 1.0f ) );

    UndirectedEdgeBitSet sharpEdges;
    genOffsetParams.outSharpEdges = &sharpEdges;

    auto resMesh = generalOffsetMesh( subMesh ? *subMesh : mp, 0.0f, genOffsetParams );
    if ( !resMesh.has_value() )
        return resMesh;

    if ( settings.decimate && resMesh->topology.numValidFaces() > 0 )
    {
        const auto map = resMesh->packOptimally( false );
        if ( !reportProgress( progress, 0.75f ) )
            return unexpectedOperationCanceled();

        sharpEdges = mapEdges( map.e, sharpEdges );

        DecimateSettings decimSettings
        {
            .maxError = 0.25f * genOffsetParams.voxelSize,
            .tinyEdgeLength = settings.tinyEdgeLength,
            .stabilizer = 1e-5f, // 1e-6 here resulted in a bit worse mesh
            .notFlippable = sharpEdges.any() ? &sharpEdges : nullptr,
            .packMesh = true,
            .progressCallback = subprogress( progress, 0.75f, 1.0f ),
            .subdivideParts = 64
        };
        if ( decimateMesh( *resMesh, decimSettings ).cancelled )
            return unexpectedOperationCanceled();
    }

    if ( settings.outSharpEdges )
        *settings.outSharpEdges = std::move( sharpEdges );

    return resMesh;
}

} //namespace MR
