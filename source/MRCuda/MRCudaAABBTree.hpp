#pragma once

#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRDipole.h"
#include <cfloat>
#include <vector>

namespace MR::Cuda
{

/// returns the nodes of given tree in the layout of CUDA kernels:
/// not-leaf nodes keep their ids, and then each leaf is a separate node with leaf id in l and invalid r;
/// leafBox( NodeId ) returns the box of a leaf
template<typename T, typename LeafBox>
std::vector<AABBTreeNode<T>> getNodesWithLeaves( const AABBTreeBase<T> & tree, LeafBox && leafBox )
{
    std::vector<AABBTreeNode<T>> res;
    if ( tree.empty() )
        return res;
    const auto & nodes = tree.nodes();
    res.reserve( 2 * nodes.size() + 1 );
    res.resize( nodes.size() );
    auto addLeaf = [&]( NodeId c )
    {
        auto & leaf = res.emplace_back();
        leaf.box = leafBox( c );
        leaf.l = NodeId( int( tree.leafId( c ) ) );
        leaf.r = NodeId();
        return NodeId( int( res.size() ) - 1 );
    };
    auto toGpu = [&]( NodeId c ) { return tree.isLeaf( c ) ? addLeaf( c ) : c; };
    if ( nodes.empty() )
    {
        addLeaf( tree.rootNodeId() );
        return res;
    }
    for ( NodeId i( 0 ); i < nodes.size(); ++i )
    {
        res[i].box = nodes[i].box;
        res[i].l = toGpu( nodes[i].l );
        res[i].r = toGpu( nodes[i].r );
    }
    return res;
}

/// the same for the trees with the boxes of leaves
template<typename T>
std::vector<AABBTreeNode<T>> getNodesWithLeaves( const AABBTreeBase<T> & tree ) requires AABBTreeBase<T>::leavesHaveBoxes
{
    return getNodesWithLeaves( tree, [&]( NodeId c ) { return tree.leaf( c ).box; } );
}

/// the same for the tree of a mesh with given coordinates of vertices
inline std::vector<AABBTreeNode<FaceTreeTraits3>> getNodesWithLeaves( const AABBTree & tree, const VertCoords & points )
{
    return getNodesWithLeaves( tree, [&]( NodeId c ) { return tree.leafBox( c, points ); } );
}

/// returns the dipoles of given tree in the layout of getNodesWithLeaves:
/// the dipoles of leaves are never good approximation, so the triangles of leaves are always computed exactly as on CPU
inline std::vector<Dipole> getDipolesWithLeaves( const AABBTreeBase<FaceTreeTraits3> & tree, const Dipoles & dipoles )
{
    std::vector<Dipole> res;
    if ( tree.empty() )
        return res;
    assert( dipoles.size() == tree.nodes().size() );
    res.reserve( 2 * dipoles.size() + 1 );
    res.assign( dipoles.begin(), dipoles.end() );
    res.resize( 2 * dipoles.size() + 1, Dipole{ .rr = FLT_MAX } );
    return res;
}

} // namespace MR::Cuda
