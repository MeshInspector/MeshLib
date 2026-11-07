#include "MRMeshProject.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRClosestPointInTriangle.h"
#include "MRBall.h"
#include "MRInplaceStack.h"
#include "MRTimer.h"
#include "MRMatrix3Decompose.h"

namespace MR
{

MeshProjectionResult findProjectionSubtree( const Vector3f & pt, const MeshPart & mp, const AABBTree & tree, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    MeshProjectionResult res;
    res.distSq = upDistLimitSq;
    if ( tree.empty() )
        return res;

    struct SubTask
    {
        NoInitNodeId n;
        float distSq;
    };
    InplaceStack<SubTask, 32> subtasks;

    auto addSubTask = [&]( const SubTask & s )
    {
        if ( !tree.isLeaf( s.n ) && s.distSq < res.distSq )
            subtasks.push( s );
    };

    NodeId bestLeaf; // without validProjections, MeshTriPoint is computed only for the best leaf in the end
    TriPointd bestBary;
    bool stop = false;
    // the triangles of leaves are tested as soon as their parent is reached, without the boxes of leaves
    auto processChild = [&]( NodeId n ) -> SubTask
    {
        if ( !tree.isLeaf( n ) )
        {
            const auto & box = tree[n].box;
            float distSq = xf ? transformed( box, *xf ).getDistanceSq( pt ) : box.getDistanceSq( pt );
            return SubTask { n, distSq };
        }
        const SubTask none{ n, FLT_MAX };
        const auto face = tree.leafId( n );
        if ( validFaces && !validFaces( face ) )
            return none;
        if ( mp.region && !mp.region->test( face ) )
            return none;
        const auto & v = tree.leafVerts( n );
        Vector3f a = mp.mesh.points[v[0]], b = mp.mesh.points[v[1]], c = mp.mesh.points[v[2]];
        if ( xf )
        {
            a = (*xf)( a );
            b = (*xf)( b );
            c = (*xf)( c );
        }

        // compute the closest point in double-precision, because float might be not enough
        const auto [projD, baryD] = closestPointInTriangle( Vector3d( pt ), Vector3d( a ), Vector3d( b ), Vector3d( c ) );
        const Vector3f proj( projD );
        const float distSq = ( proj - pt ).lengthSq();
        if ( !( distSq < res.distSq ) )
            return none;
        if ( validProjections )
        {
            const MeshProjectionResult candidate
            {
                .proj = PointOnFace{ face, proj },
                .mtp = tree.toTriPoint( mp.mesh.topology, n, TriPointf( baryD ) ),
                .distSq = distSq
            };
            if ( !validProjections( candidate ) )
                return none;
            res = candidate;
        }
        else
        {
            res.proj = PointOnFace{ face, proj };
            res.distSq = distSq;
            bestLeaf = n;
            bestBary = baryD;
        }
        if ( res.distSq <= loDistLimitSq )
            stop = true;
        return none;
    };

    addSubTask( processChild( tree.rootNodeId() ) );

    while ( !stop && !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( s.distSq >= res.distSq )
            continue;

        const auto & node = tree[s.n];
        auto s1 = processChild( node.l );
        if ( stop )
            break;
        auto s2 = processChild( node.r );
        if ( stop )
            break;
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

    if ( res.proj.face && !validProjections )
        res.mtp = tree.toTriPoint( mp.mesh.topology, bestLeaf, TriPointf( bestBary ) );
    return res;
}

MeshProjectionTransforms createProjectionTransforms( AffineXf3f& storageXf, const AffineXf3f* pointXf, const AffineXf3f* treeXf )
{
    MeshProjectionTransforms res;
    if ( treeXf && !isRigid( treeXf->A ) )
        res.nonRigidXfTree = treeXf;

    if ( res.nonRigidXfTree || !treeXf )
        res.rigidXfPoint = pointXf;
    else
    {
        storageXf = treeXf->inverse();
        if ( pointXf )
            storageXf = storageXf * ( *pointXf );
        res.rigidXfPoint = &storageXf;
    }
    return res;
}

MeshProjectionResult findProjection( const Vector3f & pt, const MeshPart & mp, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    return findProjectionSubtree( pt, mp, mp.mesh.getAABBTree(), upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
}

void findBoxedTrisInBall( const MeshPart & mp, Ball3f ball, const FoundBoxedTriCallback& foundCallback )
{
    const auto & tree = mp.mesh.getAABBTree();
    if ( tree.empty() )
        return;

    InplaceStack<NoInitNodeId, 32> subtasks;

    auto boxDistSq = [&]( NodeId n ) // squared distance from ball center to the box with interior
    {
        return tree.box( n ).getDistanceSq( ball.center );
    };

    auto addSubTask = [&]( NodeId n )
    {
        subtasks.push( n );
    };

    addSubTask( tree.rootNodeId() );

    while ( !subtasks.empty() )
    {
        const auto n = subtasks.top();
        subtasks.pop();
        if ( !( boxDistSq( n ) < ball.radiusSq ) )
            continue;

        if ( tree.isLeaf( n ) )
        {
            const auto face = tree.leafId( n );
            if ( mp.region && !mp.region->test( face ) )
                continue;
            if ( foundCallback( face, ball ) == Processing::Stop )
                break;
            continue;
        }

        const auto & node = tree[n];
        auto lDistSq = boxDistSq( node.l );
        auto rDistSq = boxDistSq( node.r );
        /// first go in the node located closer to ball's center (in case the ball will shrink and the other node will be away)
        if ( lDistSq <= rDistSq )
        {
            addSubTask( node.r );
            addSubTask( node.l );
        }
        else
        {
            addSubTask( node.l );
            addSubTask( node.r );
        }
    }
}

void findTrisInBall( const MeshPart & mp, const Ball3f & ball0, const FoundTriCallback& foundCallback, const FacePredicate & validFaces )
{
    const auto & tree = mp.mesh.getAABBTree();
    if ( tree.empty() )
        return;

    Ball3f ball = ball0; // the callback can shrink it
    InplaceStack<NoInitNodeId, 32> subtasks;
    auto addSubTask = [&]( NodeId n, float distSq )
    {
        if ( !tree.isLeaf( n ) && distSq < ball.radiusSq )
            subtasks.push( n );
    };
    bool stop = false;

    // the triangles of leaves are tested as soon as their parent is reached, without the boxes of leaves;
    // returns the squared distance from ball center to the box of a node
    auto processChild = [&]( NodeId n ) -> float
    {
        if ( !tree.isLeaf( n ) )
            return tree[n].box.getDistanceSq( ball.center );
        const auto face = tree.leafId( n );
        if ( mp.region && !mp.region->test( face ) )
            return FLT_MAX;
        if ( validFaces && !validFaces( face ) )
            return FLT_MAX;
        const auto & v = tree.leafVerts( n );
        // compute the closest point in double-precision, because float might be not enough
        const auto [projD, baryD] = closestPointInTriangle( Vector3d( ball.center ),
            Vector3d( mp.mesh.points[v[0]] ), Vector3d( mp.mesh.points[v[1]] ), Vector3d( mp.mesh.points[v[2]] ) );
        const Vector3f proj( projD );
        const auto distSq = ( proj - ball.center ).lengthSq();
        if ( !( distSq < ball.radiusSq ) )
            return FLT_MAX;
        const MeshProjectionResult candidate
        {
            .proj = PointOnFace{ face, proj },
            .mtp = tree.toTriPoint( mp.mesh.topology, n, TriPointf( baryD ) ),
            .distSq = distSq
        };
        if ( foundCallback( candidate, ball ) == Processing::Stop )
            stop = true;
        return FLT_MAX;
    };

    const auto root = tree.rootNodeId();
    addSubTask( root, processChild( root ) );

    while ( !stop && !subtasks.empty() )
    {
        const auto n = subtasks.top();
        subtasks.pop();
        if ( !( tree[n].box.getDistanceSq( ball.center ) < ball.radiusSq ) )
            continue; // the ball has shrunk

        const auto & node = tree[n];
        auto lDistSq = processChild( node.l );
        if ( stop )
            break;
        auto rDistSq = processChild( node.r );
        if ( stop )
            break;
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
