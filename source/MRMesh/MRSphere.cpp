#include "MRSphere.h"
#include "MRCube.h"
#include "MRMesh.h"
#include "MRMeshSubdivide.h"
#include "MRTimer.h"
#include <climits>

namespace MR
{

Mesh makeSphere( const SphereParams & params )
{
    MR_TIMER
    auto mesh = makeCube( Vector3f::diagonal( params.radius ), Vector3f::diagonal( -0.5f * params.radius ) );

    SubdivideSettings ss;
    ss.maxEdgeLen = params.maxEdgeLen;
    ss.maxEdgeSplits = INT_MAX;
    ss.maxDeviationAfterFlip = params.radius;
    ss.onVertCreated = [&]( VertId vid )
    { 
        mesh.points[vid] = params.radius * mesh.points[vid].normalized();
    };

    subdivideMesh( mesh, ss );

    return mesh;
}

} //namespace MR
