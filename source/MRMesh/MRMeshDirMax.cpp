#include "MRMeshDirMax.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include <cfloat>

namespace MR
{

static VertId findDirMaxBruteForce( const Vector3f & dir, const MeshPart & mp )
{
    MR_TIMER
    VertId res;
    float furthestProj = -FLT_MAX;
    if ( mp.region )
    {
        for ( auto f : *mp.region )
        {
            VertId vs[3];
            mp.mesh.topology.getTriVerts( f, vs );
            for ( auto v : vs )
            {
                auto proj = dot( mp.mesh.points[v], dir );
                if ( proj > furthestProj )
                {
                    furthestProj = proj;
                    res = v;
                }
            }
        }
    }
    else
    {
        for ( auto v : mp.mesh.topology.getValidVerts() )
        {
            auto proj = dot( mp.mesh.points[v], dir );
            if ( proj > furthestProj )
            {
                furthestProj = proj;
                res = v;
            }
        }
    }

    return res;
}

VertId findDirMax( const Vector3f & dir, const MeshPart & mp, UseAABBTree u )
{
    if ( u == UseAABBTree::No || ( u == UseAABBTree::YesIfAlreadyConstructed && !mp.mesh.getAABBTreeNotCreate() ) )
        return findDirMaxBruteForce( dir, mp );

    const AABBTree & tree = mp.mesh.getAABBTree();

    VertId res;
    if ( tree.nodes().empty() )
        return res;

    struct SubTask
    {
        NodeId n;
        float furthestBoxProj;
        SubTask() : n( noInit ) {}
        SubTask( NodeId n, float bp ) : n( n ), furthestBoxProj( bp ) { }
    };

    const Vector3f minFactor{ dir.x <= 0 ? dir.x : 0.0f, dir.y <= 0 ? dir.y : 0.0f, dir.z <= 0 ? dir.z : 0.0f };
    const Vector3f maxFactor{ dir.x >= 0 ? dir.x : 0.0f, dir.y >= 0 ? dir.y : 0.0f, dir.z >= 0 ? dir.z : 0.0f };
    auto getFurthestBoxProj = [&]( const Box3f & box )
    {
        return dot( minFactor, box.min ) + dot( maxFactor, box.max );
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;
    float furthestProj = -FLT_MAX;

    auto addSubTask = [&]( const SubTask & s )
    {
        if ( s.furthestBoxProj > furthestProj )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&]( NodeId n )
    {
        return SubTask( n, getFurthestBoxProj( tree.nodes()[n].box ) );
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto & node = tree[s.n];
        if ( s.furthestBoxProj < furthestProj )
            continue;

        if ( node.leaf() )
        {
            const auto face = node.leafId();
            if ( mp.region && !mp.region->test( face ) )
                continue;
            VertId vs[3];
            mp.mesh.topology.getTriVerts( face, vs );
            for ( int i = 0; i < 3; ++i )
            {
                auto proj = dot( mp.mesh.points[vs[i]], dir );
                if ( proj > furthestProj )
                {
                    furthestProj = proj;
                    res = vs[i];
                }
            }
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

} //namespace MR
