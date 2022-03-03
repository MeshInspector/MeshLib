#include "MRSphere.h"
#include "MRCube.h"
#include "MRMesh.h"
#include "MRMeshSubdivide.h"
#include "MRTimer.h"
#include <cmath>

namespace MR
{

Mesh makeSphere( const SphereParams & params )
{
    MR_TIMER
    auto mesh = makeCube();

    auto projectOnSphere = [&]( VertId vid )
    { 
        mesh.points[vid] = params.radius * mesh.points[vid].normalized();
    };
    for ( auto vid : mesh.topology.getValidVerts() )
        projectOnSphere( vid );

    SubdivideSettings ss;
    ss.maxEdgeSplits = params.numMeshVertices - mesh.topology.numValidVerts();
    if ( ss.maxEdgeSplits <= 0 )
        return mesh;
    ss.maxDeviationAfterFlip = params.radius;
    ss.onVertCreated = projectOnSphere;

    subdivideMesh( mesh, ss );

    return mesh;
}

} //namespace MR
