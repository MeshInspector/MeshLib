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

namespace
{

// Plain: no transformation, region, validFaces or validProjections, so the hot loop skips their checks
template <bool Plain>
MeshProjectionResult findProjectionT( const Vector3f & pt, const MeshPart & mp, const AABBTree & tree, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
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

    TriPointd resBary; // res.mtp is computed only in the end
    bool stop = false;
    // the triangles of leaves are tested as soon as their parent is reached, without the boxes of leaves
    auto processChild = [&]( NodeId n ) -> SubTask
    {
        if ( !tree.isLeaf( n ) )
        {
            const auto & box = tree.box( n );
            if constexpr ( !Plain )
                if ( xf )
                    return SubTask { n, transformed( box, *xf ).getDistanceSq( pt ) };
            return SubTask { n, box.getDistanceSq( pt ) };
        }
        const SubTask none{ n, FLT_MAX };
        const auto face = tree.leafId( n );
        const auto & v = tree.leafVerts( n );
        Vector3f a = mp.mesh.points[v[0]], b = mp.mesh.points[v[1]], c = mp.mesh.points[v[2]];
        if constexpr ( !Plain )
        {
            if ( validFaces && !validFaces( face ) )
                return none;
            if ( mp.region && !mp.region->test( face ) )
                return none;
            if ( xf )
            {
                a = (*xf)( a );
                b = (*xf)( b );
                c = (*xf)( c );
            }
        }

        // compute the closest point in double-precision, because float might be not enough
        const auto [projD, baryD] = closestPointInTriangle( Vector3d( pt ), Vector3d( a ), Vector3d( b ), Vector3d( c ) );
        const Vector3f proj( projD );
        const float distSq = ( proj - pt ).lengthSq();
        if ( !( distSq < res.distSq ) )
            return none;
        if constexpr ( !Plain )
        {
            if ( validProjections )
            {
                const MeshProjectionResult candidate
                {
                    .proj = PointOnFace{ face, proj },
                    .mtp = MeshTriPoint( mp.mesh.topology.edgeWithLeft( face ), TriPointf( baryD ) ),
                    .distSq = distSq
                };
                if ( !validProjections( candidate ) )
                    return none;
                res = candidate;
                if ( res.distSq <= loDistLimitSq )
                    stop = true;
                return none;
            }
        }
        res.proj = PointOnFace{ face, proj };
        res.distSq = distSq;
        resBary = baryD;
        if ( res.distSq <= loDistLimitSq )
            stop = true;
        return none;
    };

    // the node to visit next is not pushed in the stack
    auto s = processChild( tree.rootNodeId() );
    while ( !stop )
    {
        if ( !tree.isLeaf( s.n ) && s.distSq < res.distSq )
        {
            const auto & node = tree[s.n];
            const auto s1 = processChild( node.l );
            if ( stop )
                break;
            const auto s2 = processChild( node.r );
            if ( stop )
                break;
            // descend in the nearer child, the other waits in the stack
            const bool lNearer = s1.distSq < s2.distSq;
            addSubTask( lNearer ? s2 : s1 );
            s = lNearer ? s1 : s2;
            continue;
        }
        if ( subtasks.empty() )
            break;
        s = subtasks.top();
        subtasks.pop();
    }

    if constexpr ( !Plain )
        if ( validProjections )
            return res; // res.mtp is set with each accepted candidate
    if ( res.proj.face )
        res.mtp = MeshTriPoint( mp.mesh.topology.edgeWithLeft( res.proj.face ), TriPointf( resBary ) );
    return res;
}

} // anonymous namespace

MeshProjectionResult findProjectionSubtree( const Vector3f & pt, const MeshPart & mp, const AABBTree & tree, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    if ( !xf && !mp.region && !validFaces && !validProjections )
        return findProjectionT<true>( pt, mp, tree, upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
    return findProjectionT<false>( pt, mp, tree, upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
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
            return tree.box( n ).getDistanceSq( ball.center );
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
            .mtp = MeshTriPoint( mp.mesh.topology.edgeWithLeft( face ), TriPointf( baryD ) ),
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
        if ( !( tree.box( n ).getDistanceSq( ball.center ) < ball.radiusSq ) )
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
