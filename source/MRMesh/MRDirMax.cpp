#include "MRDirMax.h"
#include "MRSimd4.hpp"
#include "MRDirMaxBruteForce.h"
#include "MRAABBTree.h"
#include "MRAABBTreePolyline.h"
#include "MRAABBTreePoints.h"
#include "MRMesh.h"
#include "MRInplaceStack.h"
#include "MRPolyline.h"
#include "MRPointCloud.h"

#include <cfloat>
#include <type_traits>

namespace MR
{

namespace
{

// AABBTreePoints has leaf nodes, and in other trees the leaves are only referenced from nodes
template<class Tree>
constexpr bool hasLeafNodes = std::is_same_v<Tree, AABBTreePoints>;

template<class Tree>
bool emptyTree( const Tree & tree )
{
    if constexpr ( hasLeafNodes<Tree> )
        return tree.nodes().empty();
    else
        return tree.empty();
}

template<class Tree>
bool isLeaf( const Tree & tree, NodeId n )
{
    if constexpr ( hasLeafNodes<Tree> )
        return tree[n].leaf();
    else
        return tree.isLeaf( n );
}

template<class Tree>
const auto & nodeBox( const Tree & tree, NodeId n )
{
    if constexpr ( hasLeafNodes<Tree> )
        return tree[n].box;
    else
        return tree.box( n );
}

template<class V, class Tree, class LeafProcessor>
VertId findDirMaxT( const V & dir, const Tree & tree, LeafProcessor && lp )
{
    VertId res;
    if ( emptyTree( tree ) )
        return res;

    struct SubTask
    {
        NoInitNodeId n;
        float furthestBoxProj;
    };

    const auto maxCorner = Box<V>::getMaxBoxCorner( dir );

    InplaceStack<SubTask, 32> subtasks;
    float furthestProj = -FLT_MAX;

    auto addSubTask = [&]( const SubTask & s )
    {
        if ( s.furthestBoxProj > furthestProj )
            subtasks.push( s );
    };

    auto getSubTask = [&]( NodeId n )
    {
        return SubTask { n, dot( dir, nodeBox( tree, n ).corner( maxCorner ) ) };
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( s.furthestBoxProj < furthestProj )
            continue;

        if ( isLeaf( tree, s.n ) )
        {
            lp( s.n, furthestProj, res );
            continue;
        }
        const auto & node = tree[s.n];

        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
        // add task with larger projection on line last to descend there first
        if ( s1.furthestBoxProj > s2.furthestBoxProj )
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

template<class V>
VertId findDirMaxT( const V & dir, const Polyline<V> & polyline, UseAABBTree u )
{
    if ( u == UseAABBTree::No || ( u == UseAABBTree::YesIfAlreadyConstructed && !polyline.getAABBTreeNotCreate() ) )
        return findDirMaxBruteForce( dir, polyline );

    const auto & tree = polyline.getAABBTree();
    return findDirMaxT( dir, tree, [&]( NodeId n, float & furthestProj, VertId & res )
    {
        EdgeId e = tree.leafId( n );
        VertId vs[2] = { polyline.topology.org( e ), polyline.topology.dest( e ) };
        for ( int i = 0; i < 2; ++i )
        {
            auto proj = dot( polyline.points[vs[i]], dir );
            if ( proj > furthestProj )
            {
                furthestProj = proj;
                res = vs[i];
            }
        }
    } );
}

} // anonymous namespace

VertId findDirMax( const Vector3f & dir, const Mesh & m, UseAABBTree u )
{
    return findDirMax( dir, MeshPart{ m }, u );
}

namespace
{

VertId findDirMaxWide( const Vector3f & dir, const MeshPart & mp, const AABBTree & tree )
{
    using namespace Simd4;
    VertId res;
    if ( tree.nodes().empty() )
        return res;
    const auto & simdNodes = tree.simdNodes();
    const auto & simdLeaves = tree.simdLeaves();

    struct SubTask
    {
        int ref; // as in AABBTree::SimdNode::child
        float furthestBoxProj;
    };
    const auto maxCorner = Box3f::getMaxBoxCorner( dir );
    int cornerIdx[3];
    for ( int c = 0; c < 3; ++c )
        cornerIdx[c] = maxCorner[c] ? c + 3 : c;
    const F dx = set1( dir.x ), dy = set1( dir.y ), dz = set1( dir.z );
    float furthestProj = -FLT_MAX;
    // every level adds at most MaxChildren - 1 tasks, and the tree has at most MaxDepth levels
    InplaceStack<SubTask, 64> subtasks;
    subtasks.push( { tree.simdRoot(), dot( dir, tree.getBoundingBox().corner( maxCorner ) ) } );
    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( s.furthestBoxProj < furthestProj )
            continue;
        if ( s.ref < 0 )
        {
            const auto & leaf = simdLeaves[~s.ref];
            for ( int i = 0; i < leaf.count; ++i )
            {
                const auto & lt = tree.leafTris()[leaf.first + i];
                if ( mp.region && !mp.region->test( lt.face ) )
                    continue;
                for ( int j = 0; j < 3; ++j )
                {
                    const auto proj = dot( Vector3f( leaf.coord[j][0][i], leaf.coord[j][1][i], leaf.coord[j][2][i] ), dir );
                    if ( proj > furthestProj )
                    {
                        furthestProj = proj;
                        res = lt.verts[j];
                    }
                }
            }
            continue;
        }
        // the projections of the furthest corners of the children's boxes, as dot( dir, box.corner( maxCorner ) );
        // NaN or -infinity for the unused slots
        const auto & node = simdNodes[s.ref];
        alignas( 16 ) float projs[4];
        store( projs, add( add( mul( dx, load( node.box[cornerIdx[0]] ) ), mul( dy, load( node.box[cornerIdx[1]] ) ) ),
            mul( dz, load( node.box[cornerIdx[2]] ) ) ) );
        // add the child with larger projection on line last to descend there first
        SubTask ch[AABBTree::MaxChildren];
        int numCh = 0;
        for ( int i = 0; i < AABBTree::MaxChildren; ++i )
        {
            const float proj = projs[i];
            if ( !( proj > furthestProj ) )
                continue;
            int k = numCh++;
            for ( ; k > 0 && ch[k - 1].furthestBoxProj > proj; --k )
                ch[k] = ch[k - 1];
            ch[k] = { node.child[i], proj };
        }
        for ( int i = 0; i < numCh; ++i )
            subtasks.push( ch[i] );
    }
    return res;
}

} // anonymous namespace

VertId findDirMax( const Vector3f & dir, const MeshPart & mp, UseAABBTree u )
{
    if ( u == UseAABBTree::No || ( u == UseAABBTree::YesIfAlreadyConstructed && !mp.mesh.getAABBTreeNotCreate() ) )
        return findDirMaxBruteForce( dir, mp );
    return findDirMaxWide( dir, mp, mp.mesh.getAABBTree() );
}

VertId findDirMax( const Vector3f & dir, const MeshVertPart & mp, UseAABBTree u )
{
    if ( u == UseAABBTree::No || ( u == UseAABBTree::YesIfAlreadyConstructed && !mp.mesh.getAABBTreePointsNotCreate() ) )
        return findDirMaxBruteForce( dir, mp );

    return findDirMax( dir, mp.mesh.getAABBTreePoints(), mp.region );
}

VertId findDirMax( const Vector3f & dir, const Polyline3 & polyline, UseAABBTree u )
{
    return findDirMaxT( dir, polyline, u );
}

VertId findDirMax( const Vector2f & dir, const Polyline2 & polyline, UseAABBTree u )
{
    return findDirMaxT( dir, polyline, u );
}

VertId findDirMax( const Vector3f & dir, const PointCloud & cloud, const VertBitSet * region, UseAABBTree u )
{
    if ( u == UseAABBTree::No || ( u == UseAABBTree::YesIfAlreadyConstructed && !cloud.getAABBTreeNotCreate() ) )
        return findDirMaxBruteForce( dir, cloud, region );

    return findDirMax( dir, cloud.getAABBTree(), region );
}

VertId findDirMax( const Vector3f & dir, const AABBTreePoints & tree, const VertBitSet * region )
{
    const auto& orderedPoints = tree.orderedPoints();

    return findDirMaxT( dir, tree, [&]( NodeId n, float & furthestProj, VertId & res )
    {
        auto [first, last] = tree[n].getLeafPointRange();
        for ( int i = first; i < last; ++i )
        {
            if ( region )
                if ( !region->test( orderedPoints[i].id ) )
                    continue;
            auto proj = dot( orderedPoints[i].coord, dir );
            if ( proj > furthestProj )
            {
                furthestProj = proj;
                res = orderedPoints[i].id;
            }
        }
    } );
}

} //namespace MR
