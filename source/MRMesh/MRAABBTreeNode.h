#pragma once

#include "MRBox.h"
#include "MRId.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include <type_traits>

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

/// a leaf of AABB-tree: the id of an object (e.g. face) with its bounding box
template<typename T>
struct BoxedLeaf
{
    typename T::LeafId leafId;
    typename T::BoxT box;

    BoxedLeaf() noexcept = default;
    explicit BoxedLeaf( NoInit ) noexcept : leafId( noInit ), box( noInit ) { }
};

/// a leaf of the AABB-tree of a mesh: the face with the vertices of its triangle, the box of the triangle is not stored
struct FaceLeaf
{
    FaceId leafId;
    ThreeVertIds verts; ///< as returned by MeshTopology::getTriVerts when the tree was built

    FaceLeaf() noexcept = default;
    explicit FaceLeaf( NoInit ) noexcept : leafId( noInit ), verts{ VertId( noInit ), VertId( noInit ), VertId( noInit ) } { }
};

/// the type of the leaves stored in the AABB-tree with given traits
template<typename T>
struct AABBTreeLeaf { using type = BoxedLeaf<T>; };
template<>
struct AABBTreeLeaf<FaceTreeTraits3> { using type = FaceLeaf; };

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

/// a node of the AABB-tree of a mesh: only the children, the boxes of nodes and leaves are in one array AABBTree::boxes()
struct FaceTreeNode : AABBTreeChildEncoding
{
    using LeafId = FaceId;
    using BoxT = Box3f;

    NodeId l, r; ///< two children: each is either another node (non-negative id) or a leaf (encoded as negative number ~leafIndex)

    FaceTreeNode() noexcept = default;
    explicit FaceTreeNode( NoInit ) noexcept : l( noInit ), r( noInit ){}
};

/// the type of the nodes of the AABB-tree with given traits
template<typename T>
struct AABBTreeNodeType { using type = AABBTreeNode<T>; };
template<>
struct AABBTreeNodeType<FaceTreeTraits3> { using type = FaceTreeNode; };

/// whether the nodes of the tree with given traits store their boxes
template<typename T>
inline constexpr bool nodesHaveBoxes = !std::is_same_v<typename AABBTreeNodeType<T>::type, FaceTreeNode>;

template<typename T>
using AABBTreeNodeVec = Vector<typename AABBTreeNodeType<T>::type, NodeId>;

struct NodeNode
{
    NodeId aNode;
    NodeId bNode;
};

/// \}

} // namespace MR
