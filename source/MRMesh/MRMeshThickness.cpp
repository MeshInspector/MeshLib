#include "MRMeshThickness.h"
#include "MRMesh.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"
#include <cfloat>

namespace MR
{

MRMESH_API void MeshPoint::set( const Mesh& mesh, const MeshTriPoint & p )
{
    triPoint = p;
    pt = mesh.triPoint( p );
    inDir = -mesh.pseudonormal( p );
}

std::optional<MeshIntersectionResult> rayInsideIntersect( const Mesh& mesh, const MeshPoint & m )
{
    return rayMeshIntersect( mesh, { m.pt, m.inDir }, 0.0f, FLT_MAX, nullptr, true,
        [&p = m.triPoint, &top = mesh.topology]( FaceId f )
        {
            // ignore intersections with incident faces of (p)
            return !p.fromTriangle( top, f );
        } );
}

std::optional<MeshIntersectionResult> rayInsideIntersect( const Mesh& mesh, VertId v )
{
    MeshPoint m;
    m.set( mesh, MeshTriPoint( mesh.topology, v ) );
    return rayInsideIntersect( mesh, m );
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

InSphere findInCircle( const Mesh& mesh, const MeshPoint & m, const InSphereSearchSettings & settings )
{
    InSphere res;
    if ( auto isec = rayInsideIntersect( mesh, m ) )
    {
        res.center = 0.5f * ( isec->proj.point + m.pt );
        res.radius = 0.5f * isec->distanceAlongLine;
        res.oppositeTouchPoint = MeshProjectionResult{ .proj = isec->proj, .mtp = isec->mtp, .distSq = sqr( res.radius ) };
    }
    else
    {
        res.center = m.pt + m.inDir * settings.maxRadius;
        res.radius = settings.maxRadius;
        res.oppositeTouchPoint.distSq = sqr( res.radius );
    }

    for ( int it = 0; it < settings.maxRadius; ++it )
    {

    }

    return res;
}

} // namespace MR
