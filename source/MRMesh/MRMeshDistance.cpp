#include "MRMeshDistance.h"
#include "MRAABBTreeWideTraversal.hpp"
#include "MRMesh.h"
#include "MRTriDist.h"
#include "MRLine3.h"
#include "MRMeshIntersect.h"
#include "MRTimer.h"

namespace MR
{

std::optional<float> signedDistanceToMesh( const MeshPart& mp, const Vector3f& p, const SignedDistanceToMeshOptions& op )
{
    assert( op.signMode != SignDetectionMode::OpenVDB );

    auto minDistSq = op.minDistSq;
    auto maxDistSq = op.maxDistSq;
    if ( !op.nullOutsideMinMax && op.signMode == SignDetectionMode::ProjectionNormal )
    {
        // if the sign is determined by the normal at projection point then projection point must be found precisely
        minDistSq = 0;
        maxDistSq = FLT_MAX;
    }
    const auto proj = findProjection( p, mp, maxDistSq, nullptr, minDistSq );
    if ( !proj && op.signMode == SignDetectionMode::ProjectionNormal )
        return {}; // no projection point found

    if ( op.nullOutsideMinMax && ( proj.distSq < minDistSq || proj.distSq >= maxDistSq ) ) // note that proj.distSq == minDistSq (e.g. == 0) is a valid situation
        return {}; // distance is too small or too large, discard them

    float dist = std::sqrt( proj.distSq );
    switch ( op.signMode )
    {
    case SignDetectionMode::ProjectionNormal:
        if ( !mp.mesh.isOutsideByProjNorm( p, proj, mp.region ) )
            dist = -dist;
        break;

    case SignDetectionMode::OddCrossings:
    {
        const Line3d ray( Vector3d( p ), Vector3d::plusX() );
        int count = 0;
        rayMeshIntersectAll( mp, ray, [&count] ( auto&& ) { ++count; return true; } );
        if ( count % 2 == 1 ) // inside
            dist = -dist;
        break;
    }

    case SignDetectionMode::HoleWindingRule:
        assert( !mp.region );
        if ( !mp.mesh.isOutside( p, op.windingNumberThreshold, op.windingNumberBeta ) )
            dist = -dist;
        break;

    default: ; //nothing
    }
    return dist;
}

void processCloseTriangles( const MeshPart& mp, const Triangle3f & t, float rangeSq, const TriangleCallback & call )
{
    assert( call );
    if ( !call )
        return;

    const auto & tree = mp.mesh.getAABBTreeWide();
    Box3f tbox;
    for ( const auto & p : t )
        tbox.include( p );
    auto isClose = [&]( const Box3f & box ) { return box.getDistanceSq( tbox ) < rangeSq; };

    forWideTris( tree, isClose, isClose, [&]( int i )
    {
        const auto & lt = tree.leafTris()[i];
        if ( mp.region && !mp.region->test( lt.face ) )
            return Processing::Continue;
        const Triangle3f leafTriangle{ mp.mesh.points[lt.verts[0]], mp.mesh.points[lt.verts[1]], mp.mesh.points[lt.verts[2]] };
        const auto td = findTriTriDistance( t, leafTriangle, { .upDistLimitSq = rangeSq } );
        if ( td.distSq > rangeSq )
            return Processing::Continue;
        return call( td.a, lt.face, td.b, td.distSq ) == ProcessOneResult::ContinueProcessing ? Processing::Continue : Processing::Stop;
    } );
}

} //namespace MR
