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

using FaceTreeTraits3 = AABBTreeTraits<FaceTag, Box3f>;

template<typename V>
using LineTreeTraits = AABBTreeTraits<UndirectedEdgeTag, Box<V>>;
using LineTreeTraits2 = LineTreeTraits<Vector2f>;
using LineTreeTraits3 = LineTreeTraits<Vector3f>;

template<typename T>
struct AABBTreeNode
{
    using LeafId = typename T::LeafId;
    using BoxT = typename T::BoxT;

    BoxT box; ///< bounding box of whole subtree
    NodeId l, r; ///< two children: each is either another node (non-negative id) or a leaf (encoded as negative id)

    /// returns true if given child of a node is a leaf rather than another node
    [[nodiscard]] static bool isLeaf( NodeId child ) { return int( child ) < 0; }
    /// returns the id of the leaf encoded in given child of a node
    [[nodiscard]] static LeafId leafId( NodeId child ) { assert( isLeaf( child ) ); return LeafId( ~int( child ) ); }
    /// encodes given leaf as a child of a node
    [[nodiscard]] static NodeId leafChild( LeafId id ) { assert( id.valid() ); return NodeId( ~int( id ) ); }

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
