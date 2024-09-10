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
    NodeId l, r; ///< two children
    /// returns true if this is a leaf node without children nodes but with a LeafId reference
    bool leaf() const { return !r.valid(); }
    /// returns face (for the leaf node only)
    LeafId leafId() const { assert( leaf() ); return LeafId( int( l ) ); }
    void setLeafId( LeafId id ) { l = NodeId( int( id ) ); r = NodeId(); }
};

template<typename T>
using AABBTreeNodeVec = Vector<AABBTreeNode<T>, NodeId>;

/// \}

} // namespace MR
