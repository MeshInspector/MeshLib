#include "MRMeshDistance.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRTriDist.h"
#include "MRLine3.h"
#include "MRMeshIntersect.h"
#include "MRTimer.h"

namespace MR
{

std::optional<float> signedDistanceToMesh( const MeshPart& mp, const Vector3f& p, const DistanceToMeshOptions& op )
{
    assert( op.signMode != SignDetectionMode::OpenVDB );
    const auto proj = findProjection( p, mp, op.maxDistSq, nullptr, op.minDistSq );
    if ( op.signMode != SignDetectionMode::HoleWindingRule // for HoleWindingRule the sign can change even for too small or too large distances
        && ( proj.distSq < op.minDistSq || proj.distSq >= op.maxDistSq ) ) // note that proj.distSq == op.minDistSq (e.g. == 0) is a valid situation
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
        NodeId n;
        float distSq;
        SubTask() : n( noInit ) {}
        SubTask( NodeId n, float dd ) : n( n ), distSq( dd ) {}
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&]( const SubTask & s )
    {
        if ( s.distSq < rangeSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&]( NodeId n )
    {
        float distSq = tree.nodes()[n].box.getDistanceSq( tbox );
        return SubTask( n, distSq );
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
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
