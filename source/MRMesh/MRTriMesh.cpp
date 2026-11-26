#include "MRTriMesh.h"
#include "MRMesh.h"

namespace MR
{

TriMesh TriMesh::fromMesh( const Mesh & mesh )
{
    return { mesh.topology.getTriangulation(), mesh.points };
}

TriMesh TriMesh::fromMesh( Mesh && mesh )
{
    return { mesh.topology.getTriangulation(), std::move( mesh.points ) };
}

} //namespace MR
