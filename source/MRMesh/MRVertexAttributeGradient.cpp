#include "MRVertexAttributeGradient.h"
#include "MRMesh.h"
#include "MRVector.h"
#include "MRVector3.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"

namespace MR
{

Vector<Vector3f, VertId> vertexAttributeGradient( const Mesh& mesh, const Vector<float, VertId>& vertexAttribute )
{
    Vector<Vector3f, VertId> grad( mesh.topology.lastValidVert() + 1 );
    assert( vertexAttribute.size() >= grad.size() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
    {
        Vector3f sumGrad;
        int ringRank = 0;
        float orgAttrib = vertexAttribute[v];
        for ( auto e : orgRing( mesh.topology, v ) )
        {
            ++ringRank;
            sumGrad += mesh.edgeVector( e ) * ( vertexAttribute[mesh.topology.dest( e )] - orgAttrib );
        }
        sumGrad /= float( ringRank );
        grad[v] = sumGrad;
    } );
    return grad;
}

}
