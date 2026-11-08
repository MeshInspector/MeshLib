#include "MRDirMax.h"
#include "MRDirMaxBruteForce.h"
#include "MRAABBTree.h"
#include "MRAABBTreeWide.h"
#include "MRAABBTreePolyline.h"
#include "MRAABBTreePoints.h"
#include "MRMesh.h"
#include "MRInplaceStack.h"
#include "MRPolyline.h"
#include "MRPointCloud.h"

#include <cfloat>

namespace MR
{

namespace
{

template<class V, class Tree, class LeafProcessor>
VertId findDirMaxT( const V & dir, const Tree & tree, LeafProcessor && lp )
{
    VertId res;
    if ( tree.nodes().empty() )
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
        return SubTask { n, dot( dir, tree.nodes()[n].box.corner( maxCorner ) ) };
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        const auto & node = tree[s.n];
        if ( s.furthestBoxProj < furthestProj )
            continue;

        if ( node.leaf() )
        {
            lp( node, furthestProj, res );
            continue;
        }

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

    return findDirMaxT( dir, polyline.getAABBTree(), [&]( const typename AABBTreePolyline<V>::Node & node, float & furthestProj, VertId & res )
    {
        EdgeId e = node.leafId();
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

VertId findDirMaxWide( const Vector3f & dir, const MeshPart & mp, const AABBTreeWide & tree )
{
    VertId res;
    const auto & nodes = tree.nodes();
    if ( nodes.empty() )
        return res;

    struct SubTask
    {
        NoInitNodeId n;
        float furthestBoxProj;
    };
    const auto maxCorner = Box3f::getMaxBoxCorner( dir );
    float furthestProj = -FLT_MAX;
    // every level adds at most MaxChildren - 1 tasks, and the tree of 2^31 triangles has 16 levels
    InplaceStack<SubTask, 64> subtasks;
    subtasks.push( { NodeId( 0 ), dot( dir, nodes.front().box.corner( maxCorner ) ) } );
    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( s.furthestBoxProj < furthestProj )
            continue;
        const auto & node = nodes[s.n];
        if ( node.leaf() )
        {
            for ( int i = node.first, end = node.first + node.numLeafTris(); i < end; ++i )
            {
                const auto & lt = tree.leafTris()[i];
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
        SubTask ch[AABBTreeWide::MaxChildren];
        int numCh = 0;
        for ( int i = 0; i < node.count; ++i )
        {
            const NodeId n( node.first + i );
            const float proj = dot( dir, nodes[n].box.corner( maxCorner ) );
            if ( !( proj > furthestProj ) )
                continue;
            int k = numCh++;
            for ( ; k > 0 && ch[k - 1].furthestBoxProj > proj; --k )
                ch[k] = ch[k - 1];
            ch[k] = { n, proj };
        }
        for ( int i = 0; i < numCh; ++i )
            subtasks.push( ch[i] );
    }
    return res;
}

} // anonymous namespace

VertId findDirMax( const Vector3f & dir, const MeshPart & mp, UseAABBTree u )
{
    if ( u == UseAABBTree::No || ( u == UseAABBTree::YesIfAlreadyConstructed && !mp.mesh.getAABBTreeWideNotCreate() ) )
        return findDirMaxBruteForce( dir, mp );
    return findDirMaxWide( dir, mp, mp.mesh.getAABBTreeWide() );
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

    return findDirMaxT( dir, tree, [&]( const AABBTreePoints::Node & node, float & furthestProj, VertId & res )
    {
        auto [first, last] = node.getLeafPointRange();
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
