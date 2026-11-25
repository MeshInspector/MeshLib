#pragma once

#include "MRBox.h"
#include "MRId.h"
#include "MRVector2.h"
#include "MRVector3.h"

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

template<typename L, typename B>
struct MRMESH_CLASS AABBTreeTraits
{
    using LeafTag = L;
    using LeafId = Id<L>;
    using BoxT = B;
};

template<typename V>
using LineTreeTraits = AABBTreeTraits<UndirectedEdgeTag, Box<V>>;
using LineTreeTraits2 = LineTreeTraits<Vector2f>;
using LineTreeTraits3 = LineTreeTraits<Vector3f>;

/// a leaf of AABB-tree: the id of an object (e.g. face) with its bounding box
template<typename T>
struct BoxedLeaf
{
    typename T::LeafId leafId;
    typename T::BoxT box;

    BoxedLeaf() noexcept = default;
    explicit BoxedLeaf( NoInit ) noexcept : leafId( noInit ), box( noInit ) { }
};

/// the encoding of the children of AABB-tree nodes
struct AABBTreeChildEncoding
{
    /// returns true if given child of a node is a leaf rather than another node
    [[nodiscard]] static bool isLeaf( NodeId child ) { return int( child ) < 0; }
    /// returns the index in the vector of leaves of the leaf encoded in given child of a node
    [[nodiscard]] static size_t leafIndex( NodeId child ) { assert( isLeaf( child ) ); return size_t( ~int( child ) ); }
    /// encodes given index in the vector of leaves as a child of a node
    [[nodiscard]] static NodeId leafChild( size_t leafIndex ) { return NodeId( ~int( leafIndex ) ); }
};

template<typename T>
struct AABBTreeNode : AABBTreeChildEncoding
{
    using LeafId = typename T::LeafId;
    using BoxT = typename T::BoxT;

    BoxT box; ///< bounding box of whole subtree
    NodeId l, r; ///< two children: each is either another node (non-negative id) or a leaf (encoded as negative number ~leafIndex)

    AABBTreeNode() noexcept = default;
    explicit AABBTreeNode( NoInit ) noexcept : box( noInit ), l( noInit ), r( noInit ){}
};

template<typename T>
using AABBTreeNodeVec = Vector<AABBTreeNode<T>, NodeId>;

struct NodeNode
{
    NodeId aNode;
    NodeId bNode;
};

/// \}

} // namespace MR
