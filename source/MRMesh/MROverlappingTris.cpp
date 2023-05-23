#include "MROverlappingTris.h"
#include "MRMeshProject.h"
#include "MRMesh.h"
#include "MRTimer.h"

namespace MR
{

FaceBitSet findOverlappingTris( const MeshPart & mp, const FindOverlappingSettings & settings )
{
    MR_TIMER
    FaceBitSet res;
    for ( auto f : mp.mesh.topology.getFaceIds( mp.region ) )
    {
        const auto proj = findProjection( mp.mesh.triCenter( f ), mp, settings.maxDistSq, nullptr, 0, f );
        if ( !proj.proj.face || proj.mtp.bary.onEdge() >= 0 )
            continue;
        const auto normDot = dot( mp.mesh.normal( f ), mp.mesh.normal( proj.proj.face ) );
        if ( normDot > settings.maxNormalDot )
            continue;
        res.autoResizeSet( f );
        res.autoResizeSet( proj.proj.face );
    }

    return res;
}

} //namespace MR
