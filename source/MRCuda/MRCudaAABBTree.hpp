#pragma once

#include "MRCudaBasic.hpp"
#include "MRCudaMath.cuh"

#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRAABBTreePolyline.h"

#include <cstring>

namespace MR::Cuda
{

static_assert( sizeof( Box3 ) == sizeof( Box3f ) );
static_assert( sizeof( Node2 ) == sizeof( AABBTreePolyline2::Node ) );
static_assert( sizeof( LineLeaf2 ) == sizeof( AABBTreePolyline2::Leaf ) );
static_assert( sizeof( TreeNode3 ) == sizeof( AABBTreePolyline3::Node ) );
static_assert( sizeof( LineLeaf3 ) == sizeof( AABBTreePolyline3::Leaf ) );

static_assert( sizeof( SimdNode3 ) == sizeof( AABBTree::SimdNode ) );
static_assert( sizeof( LeafTri3 ) == sizeof( AABBTree::LeafTri ) );

/// the mesh's AABBTree copied to GPU as is, without conversion
struct MeshTreeBuffers
{
    DynamicArray<SimdNode3> nodes;
    DynamicArray<LeafTri3> leafTris;
    Box3 rootBox{};
    int rootRef = 0;
    int numInternal = 0;

    /// copies the internal nodes and the triangles of the leaves of not-empty tree
    cudaError_t fromTree( const AABBTree & tree )
    {
        if ( auto code = nodes.fromVector( tree.simdNodes() ) )
            return code;
        if ( auto code = leafTris.fromVector( tree.leafTris() ) )
            return code;
        const auto box = tree.getBoundingBox();
        std::memcpy( (void*)&rootBox, &box, sizeof( Box3 ) );
        rootRef = tree.rootRef();
        numInternal = int( tree.simdNodes().size() );
        return cudaSuccess;
    }

    /// the tree for the kernels
    [[nodiscard]] MeshTree3 toData() const
    {
        MeshTree3 res;
        res.nodes = nodes.data();
        res.leafTris = leafTris.data();
        res.rootBox = rootBox;
        res.rootRef = rootRef;
        res.numInternal = numInternal;
        return res;
    }

    /// the number of bytes the tree occupies on GPU
    [[nodiscard]] static size_t heapBytes( const AABBTree & tree )
    {
        return tree.simdNodes().size() * sizeof( SimdNode3 ) + tree.leafTris().size() * sizeof( LeafTri3 );
    }
};

/// the root of given not-empty tree in the encoding of CUDA kernels: 0, or ~0 if the tree has only one leaf
template<typename T>
int rootNodeId( const AABBTreeBase<T> & tree )
{
    return tree.nodes().empty() ? ~0 : 0;
}

} // namespace MR::Cuda
