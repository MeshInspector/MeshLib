#include "MRMeshDistance.h"
#include "MRAABBTree.h"
#include "MRInplaceStack.h"
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

    case SignDetectionMode::WindingRule:
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

    const AABBTree & tree = mp.mesh.getAABBTree();
    if ( tree.nodes().empty() )
        return;

    Box3f tbox;
    for ( const auto & p : t )
        tbox.include( p );

    struct SubTask
    {
        NoInitNodeId n;
        float distSq;
    };
    InplaceStack<SubTask, 32> subtasks;

    auto addSubTask = [&]( const SubTask & s )
    {
        if ( s.distSq < rangeSq )
            subtasks.push( s );
    };

    auto getSubTask = [&]( NodeId n )
    {
        float distSq = tree.nodes()[n].box.getDistanceSq( tbox );
        return SubTask { n, distSq };
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        const auto & node = tree[s.n];

        if ( node.leaf() )
        {
            const auto face = node.leafId();
            if ( mp.region && !mp.region->test( face ) )
                continue;
            const auto leafTriangle = mp.mesh.getTriPoints( face );
            Vector3f p, q;
            const float distSq = TriDist( p, q, t.data(), leafTriangle.data() );
            if ( distSq > rangeSq )
                continue;
            if ( call( p, face, q, distSq ) == ProcessOneResult::ContinueProcessing )
                continue;
            break;
        }
        
        addSubTask( getSubTask( node.r ) ); // right to look later
        addSubTask( getSubTask( node.l ) ); // left to look first
    }
}

} //namespace MR
