#include "MRAABBTreeBase.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

template <typename T>
auto AABBTreeBase<T>::getSubtrees( int minNum ) const -> std::vector<NodeId>
{
    MR_TIMER
    assert( minNum > 0 );
    std::vector<NodeId> res;
    if ( nodes_.empty() )
        return res;
    res.push_back( rootNodeId() );
    std::vector<NodeId> tmp;
    while ( res.size() < minNum )
    {
        for ( NodeId n : res )
        {
            if ( nodes_[n].leaf() )
                tmp.push_back( n );
            else
            {
                tmp.push_back( nodes_[n].l );
                tmp.push_back( nodes_[n].r );
            }
        }
        res.swap( tmp );
        if ( res.size() == tmp.size() )
            break;
        tmp.clear();
    }
    return res;
}

template <typename T>
auto AABBTreeBase<T>::getSubtreeLeaves( NodeId subtreeRoot ) const -> LeafBitSet
{
    MR_TIMER
    LeafBitSet res;

    constexpr int MaxStackSize = 32; // to avoid allocations
    NodeId subtasks[MaxStackSize];
    int stackSize = 0;
    auto addSubTask = [&]( NodeId n )
    {
        if ( nodes_[n].leaf() )
            res.autoResizeSet( nodes_[n].leafId() );
        else
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = n;
        }
    };
    addSubTask( subtreeRoot );

    while( stackSize > 0 )
    {
        NodeId n = subtasks[--stackSize];
        addSubTask( nodes_[n].r );
        addSubTask( nodes_[n].l );
    }

    return res;
}

template <typename T>
NodeBitSet AABBTreeBase<T>::getNodesFromLeaves( const LeafBitSet & leaves ) const
{
    MR_TIMER
    NodeBitSet res( nodes_.size() );

    // mark leaves
    BitSetParallelForAll( res, [&]( NodeId nid )
    {
        auto & node = nodes_[nid];
        res[nid] = node.leaf() && leaves.test( node.leafId() );
    } );

    // mark inner nodes marching from leaves to root
    for ( NodeId nid{ nodes_.size() - 1 }; nid; --nid )
    {
        auto & node = nodes_[nid];
        if ( node.leaf() )
            continue;
        res[nid] = res.test( node.l ) || res.test( node.r );
    }

    return res;
}

} //namespace MR
