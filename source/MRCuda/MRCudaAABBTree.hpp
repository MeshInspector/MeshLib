#pragma once

#include "MRCudaBasic.hpp"
#include "MRCudaMath.cuh"

#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRAABBTreePolyline.h"

namespace MR::Cuda
{

static_assert( sizeof( TreeNode3 ) == sizeof( AABBTree::Node ) );
static_assert( sizeof( FaceLeaf ) == sizeof( MR::FaceLeaf ) );
static_assert( sizeof( Box3 ) == sizeof( Box3f ) );
static_assert( sizeof( Node2 ) == sizeof( AABBTreePolyline2::Node ) );
static_assert( sizeof( LineLeaf2 ) == sizeof( AABBTreePolyline2::Leaf ) );
static_assert( sizeof( TreeNode3 ) == sizeof( AABBTreePolyline3::Node ) );
static_assert( sizeof( LineLeaf3 ) == sizeof( AABBTreePolyline3::Leaf ) );

/// the root of given not-empty tree in the encoding of CUDA kernels: 0, or ~0 if the tree has only one leaf
template<typename T>
int rootNodeId( const AABBTreeBase<T> & tree )
{
    return tree.nodes().empty() ? ~0 : 0;
}

/// AABB tree of a mesh in GPU memory as it is on CPU
struct MeshTreeBuffers
{
    DynamicArray<TreeNode3> nodes;
    DynamicArray<Box3> leafBoxes;
    DynamicArray<FaceLeaf> leaves;
    int root = 0;

    cudaError_t fromTree( const AABBTree & tree )
    {
        root = rootNodeId( tree );
        if ( auto err = nodes.fromVector( tree.nodes() ); err != cudaSuccess )
            return err;
        if ( auto err = leafBoxes.fromVector( tree.leafBoxes() ); err != cudaSuccess )
            return err;
        return leaves.fromVector( tree.leaves() );
    }

    [[nodiscard]] MeshTree data() const
    {
        return { .nodes = nodes.data(), .leafBoxes = leafBoxes.data(), .leaves = leaves.data(), .root = root };
    }

    /// the GPU memory amount required for the tree
    [[nodiscard]] static size_t heapBytes( const AABBTree & tree )
    {
        return tree.nodes().size() * sizeof( TreeNode3 ) + tree.numLeaves() * ( sizeof( Box3 ) + sizeof( FaceLeaf ) );
    }
};

} // namespace MR::Cuda
