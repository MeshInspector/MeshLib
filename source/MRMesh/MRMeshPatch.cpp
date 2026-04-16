#include "MRMeshPatch.h"
#include "MRMesh.h"
#include "MRRegionBoundary.h"
#include "MREdgePaths.h"
#include "MRTimer.h"

namespace MR
{

FaceBitSet patchMesh( Mesh& mesh, const FaceBitSet& patchBS, const FillHoleNicelySettings& settings /*= {} */ )
{
    MR_TIMER;
    FaceBitSet newFaces;
    auto bounds = delRegionKeepBd( mesh, patchBS );
    auto s = settings;
    for ( const auto& bd : bounds )
    {
        if ( bd.empty() )
            continue;
        auto avgLength = calcPathLength( bd, mesh ) / bd.size();
        if ( settings.subdivideSettings.maxEdgeLen <= 0.0f )
            s.subdivideSettings.maxEdgeLen = float( avgLength ) * 1.5f;
        if ( !mesh.topology.left( bd[0] ) )
            newFaces |= fillHoleNicely( mesh, bd[0], s );
    }
    return newFaces;
}

}