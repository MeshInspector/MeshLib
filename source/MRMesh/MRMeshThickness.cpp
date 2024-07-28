#include "MRMeshThickness.h"
#include "MRMesh.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
#include "MRClosestPointInTriangle.h"
#include "MRTimer.h"
#include <cfloat>

namespace MR
{

MRMESH_API void MeshPoint::set( const Mesh& mesh, const MeshTriPoint & p )
{
    triPoint = p;
    pt = mesh.triPoint( p );
    inDir = -mesh.pseudonormal( p );

    if ( auto v = p.inVertex( mesh.topology ) )
    {
        notIncidentFaces = [&topology=mesh.topology, v]( FaceId f )
        {
            VertId a, b, c;
            topology.getTriVerts( f, a, b, c );
            return v != a && v != b && v != c;
        };
    }
    else if ( auto oe = p.onEdge( mesh.topology ); oe && mesh.topology.right( oe.e ) )
    {
        assert( mesh.topology.left( p.e ) == mesh.topology.left( oe.e ) );
        notIncidentFaces = [f1 = mesh.topology.left( oe.e ), f2 = mesh.topology.right( oe.e )]( FaceId f )
        {
            return f1 != f && f2 != f;
        };
    }
    else
    {
        notIncidentFaces = [f1 = mesh.topology.left( p.e )]( FaceId f )
        {
            return f1 != f;
        };
    }
}

MeshIntersectionResult rayInsideIntersect( const Mesh& mesh, const MeshPoint & m, float rayEnd )
{
    return rayMeshIntersect( mesh, { m.pt, m.inDir }, 0.0f, rayEnd, nullptr, true, m.notIncidentFaces );
}

MeshIntersectionResult rayInsideIntersect( const Mesh& mesh, VertId v, float rayEnd )
{
    MeshPoint m;
    m.set( mesh, MeshTriPoint( mesh.topology, v ) );
    return rayInsideIntersect( mesh, m, rayEnd );
}

std::optional<VertScalars> computeRayThicknessAtVertices( const Mesh& mesh, const ProgressCallback & progress )
{
    MR_TIMER
    VertScalars res( mesh.points.size(), FLT_MAX );
    if ( !BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        if ( auto isec = rayInsideIntersect( mesh, v ) )
            res[v] = isec.distanceAlongLine;
    }, progress ) )
        return {};
    return res;
}

VertScalars computeThicknessAtVertices( const Mesh& mesh )
{
    return *computeRayThicknessAtVertices( mesh );
}

InSphere findInSphereImpl( const Mesh& mesh, const MeshPoint & m, const InSphereSearchSettings & settings )
{
    assert( settings.maxRadius > 0 );
    assert( settings.maxIters > 0 );
    assert( settings.minShrinkage > 0 );
    assert( settings.minShrinkage < 1 );

    // initial assessment - sphere with maximal radius
    InSphere res
    {
        .center = m.pt + m.inDir * settings.maxRadius,
        .radius = settings.maxRadius,
        .oppositeTouchPoint = { .distSq = sqr( res.radius ) }
    };

    // check candidate point, and if the sphere though it is smaller than current res, then replaces res;
    // returns true if res was updated
    auto processCandidate = [&]( const MeshProjectionResult & candidate )
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
    };

    // returns any face incident to given vertex
    auto getIncidentFace = [&mesh]( VertId d )
    {
        for ( auto ei : orgRing( mesh.topology, d ) )
            if ( auto l = mesh.topology.left( ei ) )
                return l;
        return FaceId{};
    };

    // optimization: if the point is in vertex (or on edge), check all neighbor points as candidates
    if ( auto v = m.triPoint.inVertex( mesh.topology ) )
    {
        for ( auto e : orgRing( mesh.topology, v ) )
        {
            const auto d = mesh.topology.dest( e );
            MeshProjectionResult candidate
            {
                .proj = { .point = mesh.points[d] }, // delay face setting till candidate is actually selected
                .mtp = MeshTriPoint( e.sym(), { 0, 0 } )
            };
            if ( processCandidate( candidate ) )
                res.oppositeTouchPoint.proj.face = getIncidentFace( d );
        }
    }
    else if ( auto oe = m.triPoint.onEdge( mesh.topology ) )
    {
        const auto e = oe.e;
        if ( auto l = mesh.topology.left( e ) )
        {
            const auto ne = mesh.topology.next( e );
            const auto d = mesh.topology.dest( ne );
            MeshProjectionResult candidate
            {
                .proj = { .point = mesh.points[d] }, // delay face setting till candidate is actually selected
                .mtp = MeshTriPoint( ne.sym(), { 0, 0 } )
            };
            if ( processCandidate( candidate ) )
                res.oppositeTouchPoint.proj.face = getIncidentFace( d );
        }
        if ( auto r = mesh.topology.right( e ) )
        {
            const auto pe = mesh.topology.prev( e );
            const auto d = mesh.topology.dest( pe );
            MeshProjectionResult candidate
            {
                .proj = { .point = mesh.points[d] }, // delay face setting till candidate is actually selected
                .mtp = MeshTriPoint( pe.sym(), { 0, 0 } )
            };
            if ( processCandidate( candidate ) )
                res.oppositeTouchPoint.proj.face = getIncidentFace( d );
        }
    }
    // otherwise if the point is inside a triangle, then its 3 vertices cannot touch sphere with the center located on a normal direction

    // consider ray intersection at the distance closer than the current diameter
    if ( auto isec = rayInsideIntersect( mesh, m, 2 * res.radius ) )
    {
        res.center = 0.5f * ( isec.proj.point + m.pt );
        assert( isec.distanceAlongLine >= 0 );
        res.radius = 0.5f * isec.distanceAlongLine;
        res.oppositeTouchPoint = MeshProjectionResult{ .proj = isec.proj, .mtp = isec.mtp, .distSq = sqr( res.radius ) };
    }

    findTrisInBall( mesh, Ball{ res.center, res.oppositeTouchPoint.distSq },
        [&]( MeshProjectionResult candidate, Ball & ball )
        {
            auto preRadius = res.radius;
            if ( !processCandidate( candidate ) )
                return Processing::Continue;
            if ( res.radius <= preRadius * settings.minShrinkage )
            {
                // since triangle's closest point to old sphere center is not the closest point for updated sphere center,
                // repeat several times for the same triangle
                Vector3f a, b, c;
                mesh.getTriPoints( candidate.proj.face, a, b, c );
                // start from 1 because 1 iteration was already done
                for ( int subIt = 1; subIt < settings.maxIters; ++subIt )
                {
                    preRadius = res.radius;
                    const auto [projD, baryD] = closestPointInTriangle( Vector3d( res.center ), Vector3d( a ), Vector3d( b ), Vector3d( c ) );
                    candidate.proj.point = Vector3f( projD );
                    assert( candidate.mtp.e == mesh.topology.edgeWithLeft( candidate.proj.face ) );
                    candidate.mtp.bary = TriPointf( baryD );
                    candidate.distSq = ( candidate.proj.point - res.center ).lengthSq();
                    if ( !processCandidate( candidate ) )
                        break;
                    if ( res.radius > preRadius * settings.minShrinkage )
                        break;
                }
            }
            ball = Ball{ res.center, res.oppositeTouchPoint.distSq };
            return Processing::Continue;
        }, m.notIncidentFaces );

    return res;
}

InSphere findInSphere( const Mesh& mesh, const MeshPoint & m, const InSphereSearchSettings & settings )
{
    auto res = findInSphereImpl( mesh, m, settings );
    if ( settings.insideAndOutside )
    {
        auto m2 = m;
        m2.inDir = -m2.inDir;
        auto res2 = findInSphereImpl( mesh, m2, settings );
        if ( res2.radius < res.radius )
        {
            res = res2;
            res.radius = -res.radius;
        }
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
