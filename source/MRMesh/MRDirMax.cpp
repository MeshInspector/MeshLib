#include "MRDirMax.h"
#include "MRDirMaxBruteForce.h"
#include "MRAABBTree.h"
#include "MRAABBTreePolyline.h"
#include "MRAABBTreePoints.h"
#include "MRMesh.h"
#include "MRInplaceStack.h"
#include "MRSimd4.hpp"
#include "MRPolyline.h"
#include "MRPointCloud.h"

#include <cfloat>
#include <type_traits>

namespace MR
{

namespace
{

template<class V, class Tree, class LeafProcessor>
VertId findDirMaxT( const V & dir, const Tree & tree, LeafProcessor && lp )
{
    VertId res;
    if ( tree.empty() )
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
        return SubTask { n, dot( dir, tree.box( n ).corner( maxCorner ) ) };
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( s.furthestBoxProj < furthestProj )
            continue;

        if ( tree.isLeaf( s.n ) )
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
    VertId res;
    if ( tree.empty() )
        return res;
    const auto & leafTris = tree.leafTris();

    struct SubTask
    {
        int ref; // as in AABBTree::SimdNode::child
        float furthestBoxProj;
    };
    const auto maxCorner = Box3f::getMaxBoxCorner( dir );
    // the rows of the coordinates of the corner in AABBTree::SimdBoxes
    const int cornerRow[3] = { maxCorner.x ? 3 : 0, maxCorner.y ? 4 : 1, maxCorner.z ? 5 : 2 };
    float furthestProj = -FLT_MAX;
    // every level adds at most MaxChildren - 1 tasks, and the tree has at most MaxDepth levels
    InplaceStack<SubTask, 64> subtasks;
    subtasks.push( { tree.rootRef(), dot( dir, tree.getBoundingBox().corner( maxCorner ) ) } );
    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( s.furthestBoxProj < furthestProj )
            continue;
        if ( s.ref < 0 )
        {
            for ( int i = ~s.ref * AABBTree::MaxLeafTris, end = i + AABBTree::MaxLeafTris; i < end && leafTris[i].face; ++i )
            {
                const auto & lt = leafTris[i];
                if ( mp.region && !mp.region->test( lt.face ) )
                    continue;
                for ( auto v : lt.verts )
                {
                    const auto proj = dot( mp.mesh.points[v], dir );
                    if ( proj > furthestProj )
                    {
                        furthestProj = proj;
                        res = v;
                    }
                }
            }
            continue;
        }
        // add the child with larger projection on line last to descend there first
        const auto & node = tree.simdNodes()[s.ref];
        // the projections of the corners of all children at once, the same as dot( dir, box.corner( maxCorner ) )
        alignas( 16 ) float projs[AABBTree::MaxChildren];
        {
            using namespace Simd4;
            store( projs, add( add( mul( set1( dir.x ), load( node.box[cornerRow[0]] ) ), mul( set1( dir.y ), load( node.box[cornerRow[1]] ) ) ),
                mul( set1( dir.z ), load( node.box[cornerRow[2]] ) ) ) );
        }
        SubTask ch[AABBTree::MaxChildren];
        int numCh = 0;
        for ( int i = 0; i < AABBTree::MaxChildren && node.child[i]; ++i )
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
    VertId res;
    if ( tree.empty() )
        return res;
    using namespace Simd4;
    const auto & orderedPoints = tree.orderedPoints();
    const auto & simdNodes = tree.simdNodes();

    struct SubTask
    {
        int ref; // as in AABBTreePoints::SimdNode::child
        float furthestBoxProj;
    };
    const auto maxCorner = Box3f::getMaxBoxCorner( dir );
    // the rows of the coordinates of the corner in AABBTreePoints::SimdNode
    const int cornerRow[3] = { maxCorner.x ? 3 : 0, maxCorner.y ? 4 : 1, maxCorner.z ? 5 : 2 };
    const F dx = set1( dir.x ), dy = set1( dir.y ), dz = set1( dir.z );
    float furthestProj = -FLT_MAX;
    // every level adds at most MaxChildren - 1 tasks
    InplaceStack<SubTask, 3 * AABBTreePoints::MaxDepth + 1> subtasks;
    subtasks.push( { tree.rootRef(), dot( dir, tree.getBoundingBox().corner( maxCorner ) ) } );
    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( s.furthestBoxProj < furthestProj )
            continue;
        if ( s.ref < 0 )
        {
            auto [first, last] = tree.leafPointRange( ~s.ref );
            if ( !region )
            {
                // the projections of four points at once, the same as dot( coord, dir )
                for ( ; first + 4 <= last; first += 4 )
                {
                    F r[4] = { loadu( &orderedPoints[first].coord.x ), loadu( &orderedPoints[first + 1].coord.x ),
                        loadu( &orderedPoints[first + 2].coord.x ), loadu( &orderedPoints[first + 3].coord.x ) };
                    transpose( r );
                    const F pv = add( add( mul( r[0], dx ), mul( r[1], dy ) ), mul( r[2], dz ) );
                    if ( !bits( gt( pv, set1( furthestProj ) ) ) )
                        continue;
                    alignas( 16 ) float projs[4];
                    store( projs, pv );
                    for ( int i = 0; i < 4; ++i )
                    {
                        if ( projs[i] > furthestProj )
                        {
                            furthestProj = projs[i];
                            res = orderedPoints[first + i].id;
                        }
                    }
                }
            }
            for ( int i = first; i < last; ++i )
            {
                if ( region && !region->test( orderedPoints[i].id ) )
                    continue;
                auto proj = dot( orderedPoints[i].coord, dir );
                if ( proj > furthestProj )
                {
                    furthestProj = proj;
                    res = orderedPoints[i].id;
                }
            }
            continue;
        }
        // add the child with larger projection on line last to descend there first
        const auto & node = simdNodes[s.ref];
        // the projections of the corners of all children at once, the same as dot( dir, box.corner( maxCorner ) )
        alignas( 16 ) float projs[AABBTreePoints::MaxChildren];
        store( projs, add( add( mul( dx, load( node.box[cornerRow[0]] ) ), mul( dy, load( node.box[cornerRow[1]] ) ) ),
            mul( dz, load( node.box[cornerRow[2]] ) ) ) );
        SubTask ch[AABBTreePoints::MaxChildren];
        int numCh = 0;
        for ( int i = 0; i < AABBTreePoints::MaxChildren && node.child[i]; ++i )
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

} //namespace MR
