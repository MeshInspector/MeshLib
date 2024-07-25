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

std::optional<VertScalars> computeRayThicknessAtVertices( const Mesh& mesh, const ProgressCallback & progress )
{
    MR_TIMER
    VertScalars res( mesh.points.size(), FLT_MAX );
    if ( !BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        auto isec = rayInsideIntersect( mesh, v );
        if ( isec )
            res[v] = isec->distanceAlongLine;
    }, progress ) )
        return {};
    return res;
}

VertScalars computeThicknessAtVertices( const Mesh& mesh )
{
    return *computeRayThicknessAtVertices( mesh );
}

InSphere findInSphere( const Mesh& mesh, const MeshPoint & m, const InSphereSearchSettings & settings )
{
    assert( settings.maxRadius > 0 );
    assert( settings.maxIters > 0 );
    assert( settings.minShrinkage > 0 );
    assert( settings.minShrinkage < 1 );

    InSphere res;
    if ( auto isec = rayInsideIntersect( mesh, m ) )
    {
        res.center = 0.5f * ( isec->proj.point + m.pt );
        assert( isec->distanceAlongLine >= 0 );
        res.radius = 0.5f * isec->distanceAlongLine;
        res.oppositeTouchPoint = MeshProjectionResult{ .proj = isec->proj, .mtp = isec->mtp, .distSq = sqr( res.radius ) };
    }
    else
    {
        res.center = m.pt + m.inDir * settings.maxRadius;
        res.radius = settings.maxRadius;
        res.oppositeTouchPoint.distSq = sqr( res.radius );
    }

    for ( int it = 0; it < settings.maxIters; ++it )
    {
        const auto preRadius = res.radius;
        (void)findProjection( res.center, mesh, res.oppositeTouchPoint.distSq, nullptr, 0,
            [&p = m.triPoint, &top = mesh.topology]( FaceId f )
            {
                // ignore incident faces of (p)
                return !p.fromTriangle( top, f );
            },
            [&m, &res]( const MeshProjectionResult & candidate )
            {
                const auto d = candidate.proj.point - m.pt;
                const auto dn = dot( m.inDir, d );
                if ( !( dn > 0 ) )
                    return false; // avoid circle inversion
                const auto x = sqr( d ) / ( 2 * dn );
                const auto xSq = sqr( x );
                if ( !( xSq < res.oppositeTouchPoint.distSq ) )
                    return false; // no reduction of circle
                res.center = m.pt + m.inDir * x;
                res.radius = x;
                res.oppositeTouchPoint = candidate;
                res.oppositeTouchPoint.distSq = xSq;
                return true;
            }
            );
        if ( res.radius > preRadius * settings.minShrinkage )
            break;
    }

    return res;
}

InSphere findInSphere( const Mesh& mesh, VertId v, const InSphereSearchSettings & settings )
{
    MeshPoint m;
    m.set( mesh, MeshTriPoint( mesh.topology, v ) );
    return findInSphere( mesh, m, settings );
}

std::optional<VertScalars> computeInSphereThicknessAtVertices( const Mesh& mesh, const InSphereSearchSettings & settings, const ProgressCallback & progress )
{
    MR_TIMER
    VertScalars res( mesh.points.size(), FLT_MAX );
    if ( !BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        auto sph = findInSphere( mesh, v, settings );
        res[v] = 2 * sph.radius;
    }, progress ) )
        return {};
    return res;
}

} // namespace MR
