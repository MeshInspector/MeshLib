#include "MRRebuildMesh.h"
#include "MROffset.h"
#include "MRMesh.h"
#include "MRBuffer.h"
#include "MRMapEdge.h"
#include "MRMeshDecimate.h"
#include "MRTimer.h"

namespace MR
{

Expected<Mesh> rebuildMesh( const MeshPart& mp, const RebuildMeshSettings& settings )
{
    MR_TIMER
    GeneralOffsetParameters genOffsetParams;

    genOffsetParams.voxelSize = settings.voxelSize;
    genOffsetParams.mode = settings.offsetMode;
    genOffsetParams.windingNumberThreshold = settings.windingNumberThreshold;
    genOffsetParams.windingNumberBeta = settings.windingNumberBeta;
    genOffsetParams.fwn = settings.fwn;
    genOffsetParams.callBack = subprogress( settings.progress, 0.0f, ( settings.decimate ? 0.7f : 1.0f ) );

    genOffsetParams.signDetectionMode = SignDetectionMode::HoleWindingRule;
    if ( mp.mesh.topology.isClosed( mp.region ) )
        genOffsetParams.signDetectionMode = SignDetectionMode::OpenVDB;

    UndirectedEdgeBitSet sharpEdges;
    genOffsetParams.outSharpEdges = &sharpEdges;

    auto resMesh = generalOffsetMesh( mp, 0.0f, genOffsetParams );
    if ( !resMesh.has_value() )
        return resMesh;

    if ( settings.decimate && resMesh->topology.numValidFaces() > 0 )
    {
        const auto map = resMesh->packOptimally( false );
        if ( !reportProgress( settings.progress, 0.75f ) )
            return unexpectedOperationCanceled();

        sharpEdges = mapEdges( map.e, sharpEdges );

        DecimateSettings decimSettings
        {
            .maxError = 0.25f * genOffsetParams.voxelSize,
            .tinyEdgeLength = settings.tinyEdgeLength,
            .stabilizer = 1e-5f, // 1e-6 here resulted in a bit worse mesh
            .notFlippable = sharpEdges.any() ? &sharpEdges : nullptr,
            .packMesh = true,
            .progressCallback = subprogress( settings.progress, 0.75f, 1.0f ),
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
