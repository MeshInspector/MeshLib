#include "MRMeshProject.h"
#include "MRAABBTree.h"
#include "MRAABBTreeWide.h"
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
    if ( tree.nodes().empty() )
        return res;

    struct SubTask
    {
        NoInitNodeId n;
        float distSq;
    };
    InplaceStack<SubTask, 32> subtasks;

    auto addSubTask = [&]( const SubTask & s )
    {
        if ( s.distSq < res.distSq )
            subtasks.push( s );
    };

    auto getSubTask = [&]( NodeId n )
    {
        const auto & box = tree.nodes()[n].box;
        float distSq = xf ? transformed( box, *xf ).getDistanceSq( pt ) : box.getDistanceSq( pt );
        return SubTask { n, distSq };
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
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
MeshProjectionResult findProjectionWide( const Vector3f & pt, const MeshPart & mp, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    MeshProjectionResult res;
    res.distSq = upDistLimitSq;
    const auto & tree = mp.mesh.getAABBTreeWide();
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
    auto processLeaf = [&]( const AABBTreeWide::Node & node )
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

    subtasks.push( { NodeId( 0 ), boxDistSq( NodeId( 0 ) ) } );
    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( !( s.distSq < distSq ) )
            continue;
        const auto & node = nodes[s.n];
        if ( node.leaf() )
        {
            if ( processLeaf( node ) )
                break;
            continue;
        }

        // push the children closer than the found point, the nearest last to descend there first
        SubTask ch[AABBTreeWide::MaxChildren];
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
        for ( int i = 0; i < numCh; ++i )
            subtasks.push( ch[i] );
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
    if ( !xf && !mp.region && !validFaces && !validProjections )
        return findProjectionWide<true>( pt, mp, upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
    return findProjectionWide<false>( pt, mp, upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
}

void findBoxedTrisInBall( const MeshPart & mp, Ball3f ball, const FoundBoxedTriCallback& foundCallback )
{
    const auto & tree = mp.mesh.getAABBTree();
    if ( tree.nodes().empty() )
        return;

    InplaceStack<NoInitNodeId, 32> subtasks;

    auto boxDistSq = [&]( NodeId n ) // squared distance from ball center to the box with interior
    {
        return tree.nodes()[n].box.getDistanceSq( ball.center );
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
        const auto & node = tree[n];
        if ( !( boxDistSq( n ) < ball.radiusSq ) )
            continue;

        if ( node.leaf() )
        {
            const auto face = node.leafId();
            if ( mp.region && !mp.region->test( face ) )
                continue;
            if ( foundCallback( face, ball ) == Processing::Stop )
                break;
            continue;
        }
        
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

void findTrisInBall( const MeshPart & mp, const Ball3f& ball, const FoundTriCallback& foundCallback, const FacePredicate & validFaces )
{
    findBoxedTrisInBall( mp, ball, [&]( FaceId face, Ball3f & ball )
    {
        if ( validFaces && !validFaces( face ) )
            return Processing::Continue;
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
            return foundCallback( candidate, ball );
        return Processing::Continue;
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
