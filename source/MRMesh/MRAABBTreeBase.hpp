#pragma once

#include "MRAABBTreeBase.h"
#include "MRBitSetParallelFor.h"
#include "MRInplaceStack.h"
#include "MRTimer.h"

namespace MR
{

template <typename T>
auto AABBTreeBase<T>::getSubtrees( int minNum ) const -> std::vector<NodeId>
{
    MR_TIMER;
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
    MR_TIMER;
    LeafBitSet res;

    InplaceStack<NoInitNodeId, 32> subtasks;
    auto addSubTask = [&]( NodeId n )
    {
        if ( nodes_[n].leaf() )
            res.autoResizeSet( nodes_[n].leafId() );
        else
            subtasks.push( n );
    };
    addSubTask( subtreeRoot );

    while ( !subtasks.empty() )
    {
        NodeId n = subtasks.top();
        subtasks.pop();
        addSubTask( nodes_[n].r );
        addSubTask( nodes_[n].l );
    }

    return res;
}

template <typename T>
NodeBitSet AABBTreeBase<T>::getNodesFromLeaves( const LeafBitSet & leaves ) const
{
    MR_TIMER;
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

template <typename T>
void AABBTreeBase<T>::getLeafOrder( LeafBMap & leafMap ) const
{
    MR_TIMER;
    LeafId l( 0 );
    for ( auto & n : nodes_ )
    {
        if ( !n.leaf() )
            continue;
        leafMap.b[n.leafId()] = l++;
    }
    leafMap.tsize = int( l );
}

template <typename T>
void AABBTreeBase<T>::getLeafOrderAndReset( LeafBMap & leafMap )
{
    MR_TIMER;
    LeafId l( 0 );
    for ( auto & n : nodes_ )
    {
        if ( !n.leaf() )
            continue;
        leafMap.b[n.leafId()] = l;
        n.setLeafId( l++ );
    }
    leafMap.tsize = int( l );
}

} //namespace MR
