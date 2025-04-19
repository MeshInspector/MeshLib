#include "MRDirMax.h"
#include "MRDirMaxBruteForce.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include <cfloat>

namespace MR
{

namespace
{

/// this class is intended to quickly compute maximum projection value of a box on given direction
class FurthestBoxProj
{
public:
    FurthestBoxProj( const Vector3f& dir ) :
        minFactor_{ dir.x <= 0 ? dir.x : 0.0f, dir.y <= 0 ? dir.y : 0.0f, dir.z <= 0 ? dir.z : 0.0f },
        maxFactor_{ dir.x >= 0 ? dir.x : 0.0f, dir.y >= 0 ? dir.y : 0.0f, dir.z >= 0 ? dir.z : 0.0f }
    {}

    float operator()( const Box3f & box ) const
    {
        return dot( minFactor_, box.min ) + dot( maxFactor_, box.max );
    };

private:
    Vector3f minFactor_, maxFactor_;
};

template<class Tree>
class TreeTraverser
{
public:
    TreeTraverser( const Tree & tree, const Vector3f & dir )
        : tree_( tree ), getFurthestBoxProj_( dir ) {}

    template<class LeafProcessor>
    void traverse( LeafProcessor && lp );

private:
    const Tree& tree_;
    FurthestBoxProj getFurthestBoxProj_;
    struct SubTask
    {
        NodeId n;
        float furthestBoxProj;
        SubTask() : n( noInit ) {}
        SubTask( NodeId n, float bp ) : n( n ), furthestBoxProj( bp ) { }
    };
    static constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks_[MaxStackSize];
    int stackSize_ = 0;
    float furthestProj_ = -FLT_MAX;

private:
    void addSubTask_( const SubTask & s )
    {
        if ( s.furthestBoxProj > furthestProj_ )
        {
            assert( stackSize_ < MaxStackSize );
            subtasks_[stackSize_++] = s;
        }
    };
    auto getSubTask_( NodeId n ) const
    {
        return SubTask( n, getFurthestBoxProj( tree_.nodes()[n].box ) );
    };
};

template<class Tree>
template<class LeafProcessor>
void TreeTraverser<Tree>::traverse( LeafProcessor && lp )
{
    stackSize_ = 0;
    furthestProj_ = -FLT_MAX;

    addSubTask_( getSubTask_( tree_.rootNodeId() ) );

    while( stackSize_ > 0 )
    {
        const auto s = subtasks[--stackSize_];
        const auto & node = tree[s.n];
        if ( s.furthestBoxProj < furthestProj_ )
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
}

} // anonymous namespace

VertId findDirMax( const Vector3f & dir, const MeshPart & mp, UseAABBTree u )
{
    if ( u == UseAABBTree::No || ( u == UseAABBTree::YesIfAlreadyConstructed && !mp.mesh.getAABBTreeNotCreate() ) )
        return findDirMaxBruteForce( dir, mp );

    const AABBTree & tree = mp.mesh.getAABBTree();

    VertId res;
    if ( tree.nodes().empty() )
        return res;

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
