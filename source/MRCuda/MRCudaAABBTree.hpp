#pragma once

#include "MRCudaBasic.hpp"
#include "MRCudaMath.cuh"

#include "MRMesh/MRAABBTreePolyline.h"

namespace MR::Cuda
{

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

} // namespace MR::Cuda
