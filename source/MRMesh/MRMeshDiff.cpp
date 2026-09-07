#include "MRMeshDiff.h"
#include "MRMesh.h"
#include "MRTimer.h"

namespace MR
{

MeshDiff::MeshDiff( const Mesh & from, const Mesh & to )
{
    MR_TIMER;
    pointsDiff_ = VertCoordsDiff( from.points, to.points );
    topologyDiff_ = MeshTopologyDiff( from.topology, to.topology );
}

void MeshDiff::applyAndSwap( Mesh & m )
{
    MR_TIMER;
    pointsDiff_.applyAndSwap( m.points );
    topologyDiff_.applyAndSwap( m.topology );
}

} // namespace MR
