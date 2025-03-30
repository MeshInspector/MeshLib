#include "MREdgeLengthMesh.h"
#include "MRMesh.h"
#include "MRTimer.h"

namespace MR
{

EdgeLengthMesh EdgeLengthMesh::fromMesh( const Mesh& mesh )
{
    MR_TIMER
    EdgeLengthMesh res;
    res.topology = mesh.topology;
    res.edgeLengths = mesh.edgeLengths();
    return res;
}

} //namespace MR
