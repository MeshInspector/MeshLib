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
    if ( empty() )
        return res;
    res.push_back( rootNodeId() );
    std::vector<NodeId> tmp;
    while ( res.size() < minNum )
    {
        for ( NodeId n : res )
        {
            if ( isLeaf( n ) )
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
        if ( isLeaf( n ) )
            res.autoResizeSet( leafId( n ) );
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
    if ( nodes_.empty() )
        return res;

    auto marked = [&]( NodeId c )
    {
        return isLeaf( c ) ? leaves.test( leafId( c ) ) : res.test( c );
    };

    // mark nodes marching from leaves to root (children always have larger ids than their parents)
    for ( NodeId nid = nodes_.backId(); nid >= 0; --nid )
    {
        auto & node = nodes_[nid];
        if ( marked( node.l ) || marked( node.r ) )
            res.set( nid );
    }

    return res;
}

template <typename T>
void AABBTreeBase<T>::getLeafOrder( LeafBMap & leafMap ) const
{
    MR_TIMER;
    LeafId l( 0 );
    if ( !empty() )
    {
        // depth-first traversal visits the leaves from left to right
        Vector<NodeId> subtasks; // not InplaceStack: the tree after AABBTree::addSplitFaces can be deep
        subtasks.push_back( rootNodeId() );
        while ( !subtasks.empty() )
        {
            NodeId n = subtasks.back();
            subtasks.pop_back();
            if ( isLeaf( n ) )
            {
                leafMap.b[leafId( n )] = l++;
                continue;
            }
            subtasks.push_back( nodes_[n].r );
            subtasks.push_back( nodes_[n].l );
        }
    }
    leafMap.tsize = int( l );
}

template <typename T>
void AABBTreeBase<T>::getLeafOrderAndReset( LeafBMap & leafMap )
{
    MR_TIMER;
    getLeafOrder( leafMap );
    if ( empty() )
        return;

    for ( auto & l : leaves_ )
        l.leafId = leafMap.b[l.leafId];
}

} //namespace MR
