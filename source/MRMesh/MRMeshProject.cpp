#include "MRMeshProject.h"
#include "MRSimd4.hpp"
#include "MRAABBTree.h"
#include "MRAABBTreeTraversal.hpp"
#include "MRMesh.h"
#include "MRClosestPointInTriangle.h"
#include "MRBall.h"
#include "MRInplaceStack.h"
#include "MRTimer.h"
#include "MRMatrix3Decompose.h"
#include <cmath>

namespace MR
{

MeshTriPoint MeshProjectionResult::mtp( const MeshTopology & topology ) const
{
    if ( !proj.valid() )
        return {};
    return { topology.edgeWithLeft( proj.face ), tp };
}

MeshTriPoint SignedDistanceToMeshResult::mtp( const MeshTopology & topology ) const
{
    if ( !proj.valid() )
        return {};
    return { topology.edgeWithLeft( proj.face ), tp };
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
MeshProjectionResult findProjectionWide( const Vector3f & pt, const FaceBitSet * region, const AABBTree & tree, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    using namespace Simd4;
    // the coordinates of the triangles are taken from the tree, the same as the points for which it was built or last refit
    MeshProjectionResult res;
    res.distSq = upDistLimitSq;
    if ( tree.empty() )
        return res;
    const auto & simdNodes = tree.simdNodes();
    const auto & simdLeaves = tree.simdLeaves();
    const auto & leafTris = tree.leafTris();

    // the best point so far
    float distSq = upDistLimitSq;
    FaceId face;
    Vector3f proj;
    TriPointf bary;

    struct SubTask
    {
        int ref; // as in AABBTree::SimdNode::child
        float distSq;
    };
    // every level adds at most MaxChildren - 1 tasks, and the tree has at most MaxDepth levels
    InplaceStack<SubTask, 64> subtasks;

    const F px = set1( pt.x ), py = set1( pt.y ), pz = set1( pt.z );
    // the squared distances to the children of a node
    auto childDistSq = [&]( const AABBTree::SimdNode & node, float ( &d )[4] )
    {
        if constexpr ( !Plain )
        {
            if ( xf )
            {
                for ( int i = 0; i < AABBTree::MaxChildren; ++i )
                {
                    const Box3f box( { node.box[0][i], node.box[1][i], node.box[2][i] }, { node.box[3][i], node.box[4][i], node.box[5][i] } );
                    d[i] = box.valid() ? transformed( box, *xf ).getDistanceSq( pt ) : std::numeric_limits<float>::infinity();
                }
                return;
            }
        }
        store( d, boxDistSq( node.box, px, py, pz ) );
    };
    auto rootDistSq = [&]
    {
        const auto box = tree.getBoundingBox();
        if constexpr ( !Plain )
            if ( xf )
                return transformed( box, *xf ).getDistanceSq( pt );
        return box.getDistanceSq( pt );
    };

    // the transformation is applied to the triangles in double-precision: in float, the rounding of transformed coordinates
    // distorts the shape of small triangles far from the origin, and the projections of far points on them become very inaccurate;
    // in Plain mode there is no transformation, and xfd is an empty object
    [[maybe_unused]] const auto xfd = [&]
    {
        if constexpr ( Plain )
        {
            struct NoXf {};
            return NoXf{};
        }
        else
            return xf ? AffineXf3d( *xf ) : AffineXf3d{};
    }();

    // returns true if the search can stop
    auto processLeaf = [&]( int l )
    {
        const auto & leaf = simdLeaves[l];
        // without transformation, the triangles that cannot be closer than the found point are skipped before the precise computation
        alignas( 16 ) float lowerDistSq[4];
        bool useLower = true;
        if constexpr ( !Plain )
            useLower = !xf;
        if ( useLower )
            store( lowerDistSq, triLowerDistSq( leaf.coord, px, py, pz ) );
        for ( int m = leafUsedSlots( leaf ); m; m &= m - 1 )
        {
            const int i = lowestBit( m );
            if ( useLower && !( lowerDistSq[i] < distSq ) )
                continue;
            // the triangle's ids are read only when needed: the plain search touches less memory
            auto lt = [&]() -> const AABBTree::LeafTri & { return leafTris[l * AABBTree::MaxLeafTris + i]; };
            Vector3d a( leaf.coord[0][0][i], leaf.coord[0][1][i], leaf.coord[0][2][i] );
            Vector3d b( leaf.coord[1][0][i], leaf.coord[1][1][i], leaf.coord[1][2][i] );
            Vector3d c( leaf.coord[2][0][i], leaf.coord[2][1][i], leaf.coord[2][2][i] );
            if constexpr ( !Plain )
            {
                if ( validFaces && !validFaces( lt().face ) )
                    continue;
                if ( region && !region->test( lt().face ) )
                    continue;
                if ( xf )
                {
                    a = xfd( a );
                    b = xfd( b );
                    c = xfd( c );
                }
            }
            // a triangle with a NaN coordinate has NaN distance and is never the closest (closestPointInTriangle asserts finite input)
            if ( std::isnan( a.x + a.y + a.z ) || std::isnan( b.x + b.y + b.z ) || std::isnan( c.x + c.y + c.z ) )
                continue;

            // compute the closest point in double-precision, because float might be not enough
            const auto [projD, baryD] = closestPointInTriangle( Vector3d( pt ), a, b, c );
            const Vector3f p( projD );
            const float d = ( p - pt ).lengthSq();
            if ( !( d < distSq ) )
                continue;
            if constexpr ( !Plain )
            {
                if ( validProjections && !validProjections( MeshProjectionResult
                    {
                        .proj = PointOnFace{ lt().face, p },
                        .tp = TriPointf( baryD ),
                        .distSq = d
                    } ) )
                    continue;
            }
            distSq = d;
            face = lt().face;
            proj = p;
            bary = TriPointf( baryD );
            if ( distSq <= loDistLimitSq )
                return true;
        }
        return false;
    };

    // the node to visit next is not pushed in the stack
    SubTask s{ tree.rootRef(), rootDistSq() };
    for (;;)
    {
        if ( s.distSq < distSq )
        {
            if ( s.ref < 0 )
            {
                if ( processLeaf( ~s.ref ) )
                    break;
            }
            else
            {
                const auto & node = simdNodes[s.ref];
                alignas( 16 ) float d[4];
                childDistSq( node, d );
                // the children closer than the found point ordered by distance, the farthest first
                SubTask ch[AABBTree::MaxChildren];
                int numCh = 0;
                for ( int i = 0; i < AABBTree::MaxChildren; ++i )
                {
                    if ( !( d[i] < distSq ) )
                        continue;
                    int k = numCh++;
                    for ( ; k > 0 && ch[k - 1].distSq < d[i]; --k )
                        ch[k] = ch[k - 1];
                    ch[k] = { node.child[i], d[i] };
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
        res.tp = bary;
        res.distSq = distSq;
    }
    return res;
}

} // anonymous namespace

MeshProjectionResult findProjection( const Vector3f & pt, const MeshPart & mp, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    return findProjectionSubtree( pt, mp, mp.mesh.getAABBTree(), upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
}

MeshProjectionResult findProjectionSubtree( const Vector3f & pt, const MeshPart & mp, const AABBTree & tree, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections )
{
    if ( !xf && !mp.region && !validFaces && !validProjections )
        return findProjectionWide<true>( pt, nullptr, tree, upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
    return findProjectionWide<false>( pt, mp.region, tree, upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
}

MeshProjectionResult findProjection( const Vector3f & pt, const AABBTree & tree, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq,
    const FacePredicate & validFaces, const std::function<bool(const MeshProjectionResult&)> & validProjections, const FaceBitSet * region )
{
    if ( !xf && !region && !validFaces && !validProjections )
        return findProjectionWide<true>( pt, nullptr, tree, upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
    return findProjectionWide<false>( pt, region, tree, upDistLimitSq, xf, loDistLimitSq, validFaces, validProjections );
}

namespace
{

// calls onTri( i, ball ) for the triangles (indices in leafTris()) of the tree (only from region if given) in the leaves with boxes in the ball, closer subtrees first;
// onTri can shrink the ball or stop the search
template <typename F>
void forLeafTrisInBall( const AABBTree & tree, const FaceBitSet * region, Ball3f & ball, F && onTri )
{
    if ( tree.empty() )
        return;
    const auto & leafTris = tree.leafTris();

    struct SubTask
    {
        int ref; // as in AABBTree::SimdNode::child
        Box3f box;
    };
    // every level adds at most MaxChildren - 1 nodes, and the tree has at most MaxDepth levels
    InplaceStack<SubTask, 64> subtasks;
    subtasks.push( { tree.rootRef(), tree.getBoundingBox() } );
    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( !( s.box.getDistanceSq( ball.center ) < ball.radiusSq ) )
            continue;

        if ( s.ref < 0 )
        {
            for ( int i = ~s.ref * AABBTree::MaxLeafTris, end = i + AABBTree::MaxLeafTris; i < end && leafTris[i].face; ++i )
            {
                if ( region && !region->test( leafTris[i].face ) )
                    continue;
                if ( onTri( i, ball ) == Processing::Stop )
                    return;
            }
            continue;
        }

        // first go in the nodes located closer to ball's center (in case the ball will shrink and the others will be away)
        const auto & node = tree.simdNodes()[s.ref];
        // the distances to all children at once, the same as Box3f::getDistanceSq( ball.center )
        alignas( 16 ) float d[AABBTree::MaxChildren];
        Simd4::store( d, Simd4::boxDistSq( node.box, Simd4::set1( ball.center.x ), Simd4::set1( ball.center.y ), Simd4::set1( ball.center.z ) ) );
        int ch[AABBTree::MaxChildren];
        int numCh = 0;
        for ( int i = 0; i < AABBTree::MaxChildren && node.child[i]; ++i )
        {
            if ( !( d[i] < ball.radiusSq ) )
                continue;
            int k = numCh++;
            for ( ; k > 0 && d[ch[k - 1]] < d[i]; --k )
                ch[k] = ch[k - 1];
            ch[k] = i;
        }
        for ( int k = 0; k < numCh; ++k )
            subtasks.push( { node.child[ch[k]], node.get( ch[k] ) } );
    }
}

void boxedTrisInBall( const AABBTree & tree, const FaceBitSet * region, Ball3f ball, const FoundBoxedTriCallback& foundCallback )
{
    forLeafTrisInBall( tree, region, ball, [&]( int i, Ball3f & b )
    {
        if ( !( leafTriBox( tree, i ).getDistanceSq( b.center ) < b.radiusSq ) )
            return Processing::Continue;
        return foundCallback( tree.leafTris()[i].face, b );
    } );
}

void trisInBall( const VertCoords & points, const AABBTree & tree, const FaceBitSet * region, const Ball3f& ball, const FoundTriCallback& foundCallback, const FacePredicate & validFaces )
{
    Ball3f myBall = ball;
    forLeafTrisInBall( tree, region, myBall, [&]( int i, Ball3f & b )
    {
        const auto & lt = tree.leafTris()[i];
        if ( validFaces && !validFaces( lt.face ) )
            return Processing::Continue;

        // compute the closest point in double-precision, because float might be not enough
        const auto [projD, baryD] = closestPointInTriangle( Vector3d( b.center ), Vector3d( points[lt.verts[0]] ),
            Vector3d( points[lt.verts[1]] ), Vector3d( points[lt.verts[2]] ) );
        const Vector3f proj( projD );
        const float distSq = ( proj - b.center ).lengthSq();
        if ( !( distSq < b.radiusSq ) )
            return Processing::Continue;
        const MeshProjectionResult candidate
        {
            .proj = PointOnFace{ lt.face, proj },
            .tp = TriPointf( baryD ),
            .distSq = distSq
        };
        return foundCallback( candidate, b );
    } );
}

} // anonymous namespace

void findBoxedTrisInBall( const MeshPart & mp, Ball3f ball, const FoundBoxedTriCallback& foundCallback )
{
    boxedTrisInBall( mp.mesh.getAABBTree(), mp.region, ball, foundCallback );
}

void findBoxedTrisInBall( const AABBTree & tree, Ball3f ball, const FoundBoxedTriCallback& foundCallback, const FaceBitSet * region )
{
    boxedTrisInBall( tree, region, ball, foundCallback );
}

void findTrisInBall( const MeshPart & mp, const Ball3f& ball, const FoundTriCallback& foundCallback, const FacePredicate & validFaces )
{
    trisInBall( mp.mesh.points, mp.mesh.getAABBTree(), mp.region, ball, foundCallback, validFaces );
}

void findTrisInBall( const VertCoords & points, const AABBTree & tree, const Ball3f& ball, const FoundTriCallback& foundCallback, const FacePredicate & validFaces,
    const FaceBitSet * region )
{
    trisInBall( points, tree, region, ball, foundCallback, validFaces );
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
    res->tp = projRes.tp;
    res->dist = mp.mesh.signedDistance( pt, projRes, mp.region );
    return res;
}

} //namespace MR
