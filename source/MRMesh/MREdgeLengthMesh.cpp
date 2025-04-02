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

std::optional<float> EdgeLengthMesh::edgeLengthAfterFlip( EdgeId e ) const
{
    return quadrangleOtherDiagonal(
        edgeLengths[topology.next( e )],
        edgeLengths[topology.prev( e.sym() )],
        edgeLengths[e],
        edgeLengths[topology.prev( e )],
        edgeLengths[topology.next( e.sym() )] );
}

bool EdgeLengthMesh::flipEdge( EdgeId e )
{
    const auto d = edgeLengthAfterFlip( e );
    if ( !d )
        return false;
    topology.flipEdge( e );
    edgeLengths[e] = *d;
    return true;
}

} //namespace MR
