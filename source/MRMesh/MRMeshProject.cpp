#include "MRMeshProject.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRClosestPointInTriangle.h"
#include "MRTimer.h"

namespace MR
{

MeshProjectionResult findProjectionSubtree( const Vector3f & pt, const MeshPart & mp, const AABBTree & tree, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    MeshProjectionResult res;
    res.distSq = upDistLimitSq;
    if ( tree.nodes().empty() )
        return res;

    struct SubTask
    {
        NodeId n;
        float distSq;
        SubTask() : n( noInit ) {}
        SubTask( NodeId n, float dd ) : n( n ), distSq( dd ) { }
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&]( const SubTask & s )
    {
        if ( s.distSq < res.distSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&]( NodeId n )
    {
        const auto & box = tree.nodes()[n].box;
        float distSq = xf ? transformed( box, *xf ).getDistanceSq( pt ) : box.getDistanceSq( pt );
        return SubTask( n, distSq );
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto & node = tree[s.n];
        if ( s.distSq >= res.distSq )
            continue;

        if ( node.leaf() )
        {
            const auto face = node.leafId();
            if ( validFaces && !validFaces( face ) )
                continue;
            if ( mp.region && !mp.region->test( face ) )
                continue;
            Vector3f a, b, c;
            mp.mesh.getTriPoints( face, a, b, c );
            if ( xf )
            {
                a = (*xf)( a );
                b = (*xf)( b );
                c = (*xf)( c );
            }
            
            // compute the closest point in double-precision, because float might be not enough
            const auto [projD, baryD] = closestPointInTriangle( Vector3d( pt ), Vector3d( a ), Vector3d( b ), Vector3d( c ) );
            const Vector3f proj( projD );
            const MeshProjectionResult candidate
            {
                .proj = PointOnFace{ face, proj },
                .mtp = MeshTriPoint{ mp.mesh.topology.edgeWithLeft( face ), TriPointf( baryD ) },
                .distSq = ( proj - pt ).lengthSq()
            };
            if ( validProjections && !validProjections( candidate ) )
                continue;
            if ( candidate.distSq < res.distSq )
            {
                res = candidate;
                if ( res.distSq <= loDistLimitSq )
                    break;
            }
            continue;
        }
        
        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
        // add task with smaller distance last to descend there first
        if ( s1.distSq < s2.distSq )
        {
            addSubTask( s2 );
            addSubTask( s1 );
        }
        else
        {
            addSubTask( s1 );
            addSubTask( s2 );
        }
    }

    return res;
}

MeshProjectionResult findProjection( const Vector3f & pt, const MeshPart & mp, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    return findProjectionSubtree( pt, mp, mp.mesh.getAABBTree(), upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
}

void findTrisInBall( const MeshPart & mp, Ball ball, const FoundTriCallback& foundCallback, const FacePredicate & validFaces )
{
    const auto & tree = mp.mesh.getAABBTree();
    if ( tree.nodes().empty() )
        return;

    constexpr int MaxStackSize = 32; // to avoid allocations
    NodeId subtasks[MaxStackSize];
    int stackSize = 0;

    auto boxDistSq = [&]( NodeId n ) // squared distance from ball center to the box with interior
    {
        return tree.nodes()[n].box.getDistanceSq( ball.center );
    };

    auto addSubTask = [&]( NodeId n, float boxDistSq )
    {
        if ( boxDistSq < ball.radiusSq ) // ball intersects the box
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = n;
        }
    };

    addSubTask( tree.rootNodeId(), boxDistSq( tree.rootNodeId() ) );

    while( stackSize > 0 )
    {
        const auto n = subtasks[--stackSize];
        const auto & node = tree[n];
        if ( !( boxDistSq( n ) < ball.radiusSq ) ) // check again in case the ball has changed
            continue;

        if ( node.leaf() )
        {
            const auto face = node.leafId();
            if ( validFaces && !validFaces( face ) )
                continue;
            if ( mp.region && !mp.region->test( face ) )
                continue;
            Vector3f a, b, c;
            mp.mesh.getTriPoints( face, a, b, c );
            
            // compute the closest point in double-precision, because float might be not enough
            const auto [projD, baryD] = closestPointInTriangle( Vector3d( ball.center ), Vector3d( a ), Vector3d( b ), Vector3d( c ) );
            const Vector3f proj( projD );
            const MeshProjectionResult candidate
            {
                .proj = PointOnFace{ face, proj },
                .mtp = MeshTriPoint{ mp.mesh.topology.edgeWithLeft( face ), TriPointf( baryD ) },
                .distSq = ( proj - ball.center ).lengthSq()
            };
            if ( candidate.distSq < ball.radiusSq )
            {
                if ( foundCallback( candidate, ball ) == Processing::Stop )
                    break;
            }
            continue;
        }
        
        auto lDistSq = boxDistSq( node.l );
        auto rDistSq = boxDistSq( node.r );
        /// first go in the node located closer to ball's center (in case the ball will shrink and the other node will be away)
        if ( lDistSq <= rDistSq )
        {
            addSubTask( node.r, rDistSq );
            addSubTask( node.l, lDistSq );
        }
        else
        {
            addSubTask( node.l, lDistSq );
            addSubTask( node.r, rDistSq );
        }
    }
}

std::optional<SignedDistanceToMeshResult> findSignedDistance( const Vector3f & pt, const MeshPart & mp,
    float upDistLimitSq, float loDistLimitSq )
{
    auto projRes = findProjection( pt, mp, upDistLimitSq, nullptr, loDistLimitSq );
    std::optional<SignedDistanceToMeshResult> res;
    if ( !( projRes.distSq < upDistLimitSq ) || projRes.distSq < loDistLimitSq )
    {
        return res;
    }
    res = SignedDistanceToMeshResult();
    res->proj = projRes.proj;
    res->mtp = projRes.mtp;
    res->dist = mp.mesh.signedDistance( pt, projRes, mp.region );
    return res;
}

} //namespace MR
