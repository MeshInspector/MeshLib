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

    auto projectOnSphere = [radius = params.radius]( const Vector3f & v )
    { 
        return radius * v.normalized();
    };
    for ( auto & p : mesh.points )
        p = projectOnSphere( p );

    SubdivideSettings ss;
    ss.maxEdgeLen = params.maxEdgeLen;

    // sphere area is 4*pi*radius^2
    // min triangle area 1/2*(1/4*maxEdgeLen^2)
    // every split create 2 triangles
    // maxSplits = 4*pi*radius^2 / (1/4*maxEdgeLen^2), but we put a larger limit
    ss.maxEdgeSplits = (int)std::lround( sqr( 15 * params.radius / params.maxEdgeLen ) );

    ss.maxDeviationAfterFlip = params.radius;
    ss.onVertCreated = [&]( VertId vid )
    { 
        mesh.points[vid] = projectOnSphere( mesh.points[vid] );
    };

    subdivideMesh( mesh, ss );

    return mesh;
}

} //namespace MR
