#include "MRMeshDelete.h"
#include "MRMeshNormals.h"
#include "MRMesh.h"
#include "MRVector.h"
#include "MRTimer.h"
#include "MRMeshBuilder.h"

namespace MR
{

void deleteTargetFaces( Mesh& obj, const Vector3f& targetCenter )
{
    MR_TIMER;
    MR_WRITER( obj );

    auto& topology = obj.topology;
    auto& edgePerFaces = topology.edgePerFace();
    auto& points = obj.points;
    for ( FaceId i{ 0 }; i < edgePerFaces.size(); ++i )
    {
        auto& edge = edgePerFaces[i];
        if ( !edge.valid() )
            continue;
        VertId v0, v1, v2;
        topology.getLeftTriVerts( edge, v0, v1, v2 );
        auto norm = cross( points[v1] - points[v0], points[v2] - points[v0] );
        auto center = ( points[v2] + points[v1] + points[v0] ) / 3.f;
        auto v = dot( norm, targetCenter - center );
        if ( v > 0.f )
        {
            topology.deleteFace( i );
        }
    }
}

void deleteTargetFaces( Mesh & obj, const Mesh & target )
{
    MR_TIMER;
    MR_WRITER( obj );

    // lets find the center of the tooth root
    Vector3f targetCenter = target.findCenterFromFaces();
    deleteTargetFaces( obj, targetCenter );
}

} //namespace MR
