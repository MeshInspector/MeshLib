#include "MROffsetVerts.h"

#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

void offsetVerts( Mesh& mesh, const VertMetric& offset, ProgressCallback cb )
{
    MR_TIMER;
    BitSetParallelFor( mesh.topology.getValidVerts(), [&offset, &mesh] ( VertId v )
    {
        mesh.points[v] += mesh.pseudonormal( v ) * offset( v );
    }, cb );
}

}

