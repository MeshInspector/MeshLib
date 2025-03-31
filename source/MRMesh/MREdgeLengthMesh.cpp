#include "MREdgeLengthMesh.h"
#include "MRMesh.h"
#include "MRTriMath.h"
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

float EdgeLengthMesh::leftCotan( EdgeId e ) const
{
    if ( !topology.left( e ).valid() )
        return 0;

    EdgeId e1, e2;
    topology.getLeftTriEdges( e, e1, e2 );
    return MR::cotan( edgeLengths[e], edgeLengths[e1], edgeLengths[e2] );
}

} //namespace MR
