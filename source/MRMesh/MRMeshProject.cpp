#include "MRMeshProject.h"
#include "MRAABBTree.h"
#include "MRAABBTreeTraversal.hpp"
#include "MRMesh.h"
#include "MRClosestPointInTriangle.h"
#include "MRBall.h"
#include "MRInplaceStack.h"
#include "MRTimer.h"
#include "MRMatrix3Decompose.h"

namespace MR
{

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

namespace
{

// Plain: no transformation, region, validFaces or validProjections, so the hot loops skip their checks
template <bool Plain>
MeshProjectionResult findProjectionWide( const Vector3f & pt, const MeshPart & mp, const AABBTree & tree, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    MeshProjectionResult res;
    res.distSq = upDistLimitSq;
    const auto & nodes = tree.nodes();
    if ( nodes.empty() )
        return res;

    // the best point so far
    float distSq = upDistLimitSq;
    FaceId face;
    Vector3f proj;
    TriPointf bary;

    struct SubTask
    {
        NoInitNodeId n;
        float distSq;
    };
    // every level adds at most MaxChildren - 1 tasks, and the tree of 2^31 triangles has 16 levels
    InplaceStack<SubTask, 64> subtasks;

    auto boxDistSq = [&]( NodeId n )
    {
        const auto & box = nodes[n].box;
        if constexpr ( !Plain )
            if ( xf )
                return transformed( box, *xf ).getDistanceSq( pt );
        return box.getDistanceSq( pt );
    };

    // returns true if the search can stop
    auto processLeaf = [&]( const AABBTree::Node & node )
    {
        for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
        {
            const auto & lt = tree.leafTris()[i];
            Vector3f a = mp.mesh.points[lt.verts[0]];
            Vector3f b = mp.mesh.points[lt.verts[1]];
            Vector3f c = mp.mesh.points[lt.verts[2]];
            if constexpr ( !Plain )
            {
                if ( validFaces && !validFaces( lt.face ) )
                    continue;
                if ( mp.region && !mp.region->test( lt.face ) )
                    continue;
                if ( xf )
                {
                    a = (*xf)( a );
                    b = (*xf)( b );
                    c = (*xf)( c );
                }
            }

            // compute the closest point in double-precision, because float might be not enough
            const auto [projD, baryD] = closestPointInTriangle( Vector3d( pt ), Vector3d( a ), Vector3d( b ), Vector3d( c ) );
            const Vector3f p( projD );
            const float d = ( p - pt ).lengthSq();
            if ( !( d < distSq ) )
                continue;
            if constexpr ( !Plain )
            {
                if ( validProjections && !validProjections( MeshProjectionResult
                    {
                        .proj = PointOnFace{ lt.face, p },
                        .mtp = MeshTriPoint{ mp.mesh.topology.edgeWithLeft( lt.face ), TriPointf( baryD ) },
                        .distSq = d
                    } ) )
                    continue;
            }
            distSq = d;
            face = lt.face;
            proj = p;
            bary = TriPointf( baryD );
            if ( distSq <= loDistLimitSq )
                return true;
        }
        return false;
    };

    // the node to visit next is not pushed in the stack
    SubTask s{ NodeId( 0 ), boxDistSq( NodeId( 0 ) ) };
    for (;;)
    {
        if ( s.distSq < distSq )
        {
            const auto & node = nodes[s.n];
            if ( node.leaf() )
            {
                if ( processLeaf( node ) )
                    break;
            }
            else
            {
                // the children closer than the found point ordered by distance, the farthest first
                SubTask ch[AABBTree::MaxChildren];
                int numCh = 0;
                for ( int i = 0; i < node.count; ++i )
                {
                    const NodeId n( node.first + i );
                    const float d = boxDistSq( n );
                    if ( !( d < distSq ) )
                        continue;
                    int k = numCh++;
                    for ( ; k > 0 && ch[k - 1].distSq < d; --k )
                        ch[k] = ch[k - 1];
                    ch[k] = { n, d };
                }
                if ( numCh > 0 )
                {
                    // descend in the nearest child, the others wait in the stack
                    for ( int i = 0; i + 1 < numCh; ++i )
                        subtasks.push( ch[i] );
                    s = ch[numCh - 1];
                    continue;
                }
            }
        }
        if ( subtasks.empty() )
            break;
        s = subtasks.top();
        subtasks.pop();
    }

    if ( face )
    {
        res.proj = PointOnFace{ face, proj };
        res.mtp = MeshTriPoint{ mp.mesh.topology.edgeWithLeft( face ), bary };
        res.distSq = distSq;
    }
    return res;
}

} // anonymous namespace

MeshProjectionResult findProjection( const Vector3f & pt, const MeshPart & mp, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    const auto & tree = mp.mesh.getAABBTree();
    if ( !xf && !mp.region && !validFaces && !validProjections )
        return findProjectionWide<true>( pt, mp, tree, upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
    return findProjectionWide<false>( pt, mp, tree, upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
}

MeshProjectionResult findProjectionSubtree( const Vector3f & pt, const MeshPart & mp, const AABBTree & tree, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    return findProjectionWide<false>( pt, mp, tree, upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
}

namespace
{

// calls onTri( i, ball ) for the triangles (indices in leafTris()) of mp in the leaves with boxes in the ball, closer subtrees first;
// onTri can shrink the ball or stop the search
template <typename F>
void forLeafTrisInBall( const MeshPart & mp, Ball3f & ball, F && onTri )
{
    const auto & tree = mp.mesh.getAABBTree();
    const auto & nodes = tree.nodes();
    if ( nodes.empty() )
        return;

    // every level adds at most MaxChildren - 1 nodes, and the tree of 2^31 triangles has 16 levels
    InplaceStack<NoInitNodeId, 64> subtasks;
    subtasks.push( NodeId( 0 ) );
    while ( !subtasks.empty() )
    {
        const auto & node = nodes[subtasks.top()];
        subtasks.pop();
        if ( !( node.box.getDistanceSq( ball.center ) < ball.radiusSq ) )
            continue;

        if ( node.leaf() )
        {
            for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
            {
                if ( mp.region && !mp.region->test( tree.leafTris()[i].face ) )
                    continue;
                if ( onTri( i, ball ) == Processing::Stop )
                    return;
            }
            continue;
        }

        // first go in the nodes located closer to ball's center (in case the ball will shrink and the others will be away)
        NodeId ch[AABBTree::MaxChildren];
        float chDistSq[AABBTree::MaxChildren];
        int numCh = 0;
        for ( int i = 0; i < node.count; ++i )
        {
            const NodeId n( node.first + i );
            const float distSq = nodes[n].box.getDistanceSq( ball.center );
            if ( !( distSq < ball.radiusSq ) )
                continue;
            int k = numCh++;
            for ( ; k > 0 && chDistSq[k - 1] < distSq; --k )
            {
                ch[k] = ch[k - 1];
                chDistSq[k] = chDistSq[k - 1];
            }
            ch[k] = n;
            chDistSq[k] = distSq;
        }
        for ( int i = 0; i < numCh; ++i )
            subtasks.push( ch[i] );
    }
}

} // anonymous namespace

void findBoxedTrisInBall( const MeshPart & mp, Ball3f ball, const FoundBoxedTriCallback& foundCallback )
{
    const auto & tree = mp.mesh.getAABBTree();
    forLeafTrisInBall( mp, ball, [&]( int i, Ball3f & b )
    {
        if ( !( tree.leafBoxes()[i].getDistanceSq( b.center ) < b.radiusSq ) )
            return Processing::Continue;
        return foundCallback( tree.leafTris()[i].face, b );
    } );
}

void findTrisInBall( const MeshPart & mp, const Ball3f& ball, const FoundTriCallback& foundCallback, const FacePredicate & validFaces )
{
    Ball3f myBall = ball;
    const auto & tree = mp.mesh.getAABBTree();
    forLeafTrisInBall( mp, myBall, [&]( int i, Ball3f & b )
    {
        const auto & lt = tree.leafTris()[i];
        if ( validFaces && !validFaces( lt.face ) )
            return Processing::Continue;

        // compute the closest point in double-precision, because float might be not enough
        const auto [projD, baryD] = closestPointInTriangle( Vector3d( b.center ), Vector3d( mp.mesh.points[lt.verts[0]] ),
            Vector3d( mp.mesh.points[lt.verts[1]] ), Vector3d( mp.mesh.points[lt.verts[2]] ) );
        const Vector3f proj( projD );
        const float distSq = ( proj - b.center ).lengthSq();
        if ( !( distSq < b.radiusSq ) )
            return Processing::Continue;
        const MeshProjectionResult candidate
        {
            .proj = PointOnFace{ lt.face, proj },
            .mtp = MeshTriPoint{ mp.mesh.topology.edgeWithLeft( lt.face ), TriPointf( baryD ) },
            .distSq = distSq
        };
        return foundCallback( candidate, b );
    } );
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
