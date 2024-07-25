#include "MRMeshThickness.h"
#include "MRMesh.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"
#include <cfloat>

namespace MR
{

std::optional<MeshIntersectionResult> rayInsideIntersect( const Mesh& mesh, const MeshTriPoint & p )
{
    const auto dir = -mesh.pseudonormal( p );
    return rayMeshIntersect( mesh, { mesh.triPoint( p ), dir }, 0.0f, FLT_MAX, nullptr, true,
        [&p, &top = mesh.topology]( FaceId f )
        {
            // ignore intersections with incident faces of (p)
            return !p.fromTriangle( top, f );
        } );
}

std::optional<MeshIntersectionResult> rayInsideIntersect( const Mesh& mesh, VertId v )
{
    return rayInsideIntersect( mesh, MeshTriPoint( mesh.topology, v ) );
}

VertScalars computeRayThicknessAtVertices( const Mesh& mesh )
{
    MR_TIMER
    VertScalars res( mesh.points.size(), FLT_MAX );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        auto isec = rayInsideIntersect( mesh, v );
        if ( isec )
            res[v] = isec->distanceAlongLine;
    } );
    return res;
}

VertScalars computeThicknessAtVertices( const Mesh& mesh )
{
    return computeRayThicknessAtVertices( mesh );
}

} // namespace MR
