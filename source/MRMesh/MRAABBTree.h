#pragma once

#include "MRBox.h"
#include "MRId.h"
#include "MRVector.h"
#include "MRResizeNoInit.h"
#include "MRMeshPart.h"
#include <algorithm>
#include <limits>
#include <vector>

namespace MR
{

/**
 * \defgroup AABBTreeGroup AABB Tree overview
 * \brief This chapter represents documentation about AABB Tree
 */

/// \addtogroup AABBTreeGroup
/// \{

/// triangle in a leaf of AABBTree
struct AABBTreeWideLeafTri
{
    FaceId face; ///< invalid in the unused slots of a leaf
    ThreeVertIds verts; ///< in the order of mesh.topology.getTriVerts( face )

    AABBTreeWideLeafTri() = default;
    AABBTreeWideLeafTri( FaceId face, const ThreeVertIds & verts ) noexcept : face( face ), verts( verts ) {}
    /// all members uninitialized, e.g. for the triangles written by the tree builder
    explicit AABBTreeWideLeafTri( NoInit ) noexcept : face( noInit ), verts{ VertId( noInit ), VertId( noInit ), VertId( noInit ) } {}
};

/// four boxes in SoA order: [0..2][i] - min x,y,z of i-th box, [3..5][i] - its max x,y,z;
/// the unused slots have empty boxes (min = +inf, max = -inf)
struct MR_BIND_IGNORE AABBTreeSimdBoxes
{
    alignas( 16 ) float box[6][4];

    AABBTreeSimdBoxes() = default;
    explicit AABBTreeSimdBoxes( NoInit ) noexcept {}

    /// returns i-th box
    [[nodiscard]] Box3f get( int i ) const { return { { box[0][i], box[1][i], box[2][i] }, { box[3][i], box[4][i], box[5][i] } }; }

    /// sets i-th box
    void set( int i, const Box3f & b )
    {
        for ( int c = 0; c < 3; ++c )
        {
            box[c][i] = b.min[c];
            box[c + 3][i] = b.max[c];
        }
    }

    /// makes i-th box empty
    void setEmpty( int i )
    {
        constexpr float inf = std::numeric_limits<float>::infinity();
        for ( int c = 0; c < 3; ++c )
        {
            box[c][i] = inf;
            box[c + 3][i] = -inf;
        }
    }

    /// returns the union of all four boxes (as Box3f::include of them in their order)
    [[nodiscard]] Box3f getUnion() const
    {
        Box3f res;
        for ( int i = 0; i < 4; ++i )
            res.include( get( i ) );
        return res;
    }
};

/// internal node of AABBTree: the boxes of its (up to 4) children in SoA order and the references to them;
/// lets a traversal test all children of a node at once
struct MR_BIND_IGNORE AABBTreeSimdNode : AABBTreeSimdBoxes
{
    /// i-th child: the index in simdNodes() if > 0, otherwise ~(the index in simdLeaves()); 0 in unused slots (the root is never a child)
    int child[4];

    AABBTreeSimdNode() = default;
    explicit AABBTreeSimdNode( NoInit ) noexcept : AABBTreeSimdBoxes( noInit ) {}
};

/// leaf of AABBTree: the vertex coordinates of its (up to 4) triangles in SoA order; lets a traversal test all triangles of a leaf at once
struct MR_BIND_IGNORE AABBTreeSimdLeaf
{
    /// [v][c][i] - coordinate c of vertex v (in the order of LeafTri::verts) of i-th triangle; NaN in unused slots
    alignas( 16 ) float coord[3][3][4];

    AABBTreeSimdLeaf() = default;
    explicit AABBTreeSimdLeaf( NoInit ) noexcept {}

    /// sets the coordinates of the vertices of i-th triangle
    void set( int i, const Vector3f & a, const Vector3f & b, const Vector3f & c )
    {
        const Vector3f * p[3] = { &a, &b, &c };
        for ( int v = 0; v < 3; ++v )
            for ( int k = 0; k < 3; ++k )
                coord[v][k][i] = ( *p[v] )[k];
    }

    /// makes i-th slot unused: NaN coordinates
    void setEmpty( int i )
    {
        constexpr float nan = std::numeric_limits<float>::quiet_NaN();
        for ( int v = 0; v < 3; ++v )
            for ( int k = 0; k < 3; ++k )
                coord[v][k][i] = nan;
    }
};

/// bounding volume hierarchy of mesh triangles with up to 4 children per node and up to 4 triangles per leaf
/// in SIMD layout: every internal node keeps the boxes of its children, and every leaf the coordinates of its triangles;
/// the triangles of leaf l are in slots [4 * l, 4 * l + 4) of leafTris(), its unused slots have invalid faces;
/// a node is referenced as in AABBTreeSimdNode::child, or by NodeId: the internal nodes first, then the leaves
class AABBTree
{
public:
    static constexpr int MaxChildren = 4;
    static constexpr int MaxLeafTris = 4;
    /// the deepest level of a node (the root is at 0): the stacks of the traversals are sized for it, and addSplitFaces rebuilds a deeper tree
    static constexpr int MaxDepth = 16;

    using LeafTri = AABBTreeWideLeafTri;
    using SimdBoxes = AABBTreeSimdBoxes;
    using SimdNode = AABBTreeSimdNode;
    using SimdLeaf = AABBTreeSimdLeaf;

    /// creates tree for given mesh or its part
    [[nodiscard]] MRMESH_API explicit AABBTree( const MeshPart & mp );

    /// creates tree for the triangles of tmesh (all if region is nullptr), faster than from a mesh;
    /// it is the tree of a mesh with tmesh.points and mesh.topology.getTriVerts( f ) == tmesh.tris[f] for all its triangles
    [[nodiscard]] MRMESH_API explicit AABBTree( const TriMesh & tmesh, const FaceBitSet * region = nullptr );

    AABBTree() = default;
    AABBTree( AABBTree && ) noexcept = default;
    AABBTree & operator =( AABBTree && ) noexcept = default;

    /// returns true if the tree has no triangles
    [[nodiscard]] bool empty() const { return simdLeaves_.empty(); }

    /// returns the number of triangles in the tree
    [[nodiscard]] int numTris() const { return numTris_; }

    /// the internal nodes, the root is the first one (if the tree has more than MaxLeafTris triangles)
    [[nodiscard]] MR_BIND_IGNORE const Vector<SimdNode> & simdNodes() const { return simdNodes_; }

    /// the leaves
    [[nodiscard]] MR_BIND_IGNORE const Vector<SimdLeaf> & simdLeaves() const { return simdLeaves_; }

    /// the bounding boxes of the triangles of given leaf, slightly expanded for robust searches, computed from its coordinates
    [[nodiscard]] MR_BIND_IGNORE MRMESH_API SimdBoxes leafBoxes( int leaf ) const;

    /// the triangles of all leaves, MaxLeafTris slots per leaf
    [[nodiscard]] const Vector<LeafTri> & leafTris() const { return leafTris_; }

    /// the bounding box of the triangle in slot i of leafTris(), slightly expanded for robust searches, computed from its coordinates (empty for an unused slot)
    [[nodiscard]] MRMESH_API Box3f leafBox( int i ) const;

    /// the coordinates of vertex v (in the order of LeafTri::verts) of the triangle in slot i of leafTris(), the same as the points of the mesh
    [[nodiscard]] Vector3f leafPoint( int i, int v ) const
    {
        const auto & c = simdLeaves_[i / MaxLeafTris].coord[v];
        const int k = i % MaxLeafTris;
        return { c[0][k], c[1][k], c[2][k] };
    }

    /// the number of triangles in given leaf
    [[nodiscard]] int numLeafTris( int leaf ) const
    {
        const auto * lt = leafTris_.data() + leaf * MaxLeafTris;
        return 1 + lt[1].face.valid() + lt[2].face.valid() + lt[3].face.valid();
    }

    /// the reference to the root as in SimdNode::child: 0, or ~0 if the tree has one leaf only
    [[nodiscard]] int rootRef() const { return simdNodes_.empty() ? ~0 : 0; }

    /// returns the bounding box of the whole tree (empty for an empty tree)
    [[nodiscard]] Box3f getBoundingBox() const
        { return !simdNodes_.empty() ? simdNodes_.front().getUnion() : !simdLeaves_.empty() ? leafBoxes( 0 ).getUnion() : Box3f{}; }

    /// the total number of nodes: internal ones and leaves
    [[nodiscard]] size_t numNodes() const { return simdNodes_.size() + simdLeaves_.size(); }

    /// the id of the node with given reference (as in SimdNode::child): the internal nodes first, then the leaves
    [[nodiscard]] NodeId nodeId( int ref ) const { return NodeId( ref >= 0 ? ref : int( simdNodes_.size() ) + ~ref ); }

    /// the reference (as in SimdNode::child) to the node with given id
    [[nodiscard]] int nodeRef( NodeId n ) const { return int( n ) < int( simdNodes_.size() ) ? int( n ) : ~( int( n ) - int( simdNodes_.size() ) ); }

    /// returns the bounding boxes of all nodes
    [[nodiscard]] MRMESH_API Vector<Box3f, NodeId> getNodeBoxes() const;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return simdNodes_.capacity() * sizeof( SimdNode ) + simdLeaves_.capacity() * sizeof( SimdLeaf )
        + leafTris_.heapBytes(); }

    /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
    [[nodiscard]] MRMESH_API std::vector<NodeId> getSubtrees( int minNum ) const;

    /// returns all triangles in the subtree with given root
    [[nodiscard]] MRMESH_API FaceBitSet getSubtreeFaces( NodeId subtreeRoot ) const;

    /// returns the nodes having at least one of given triangles in their subtrees
    [[nodiscard]] MRMESH_API NodeBitSet getNodesFromFaces( const FaceBitSet & faces ) const;

    /// fills map: FaceId -> its position in the order of the triangles in the leaves;
    /// buffer in faceMap must be resized before the call, and caller is responsible for filling missing face elements
    MRMESH_API void getLeafOrder( FaceBMap & faceMap ) const;

    /// renumbers the triangles and their vertices after the mesh was packed with given maps from old to new ids
    MRMESH_API void renumber( const FaceBMap & faceMap, const VertBMap & vertMap );

    /// changes the order of the vertices of the triangles of given faces (all if nullptr) as MeshTopology::flipOrientation() does;
    /// shall be called after MeshTopology::flipOrientation() if this tree is kept
    MRMESH_API void flipOrientation( const FaceBitSet * region = nullptr );

    /// updates the boxes of the triangles with changed vertices and of their ancestors;
    /// this is a faster alternative to full tree rebuild (but the tree after refit might be less efficient)
    /// \param mesh same mesh for which this tree was constructed but with updated coordinates;
    /// \param changedVerts vertex ids with modified coordinates (since tree construction or last refit), nullptr means all vertices
    MRMESH_API void refit( const Mesh & mesh, const VertBitSet * changedVerts );

    /// adds to the tree the faces that appeared after splitting some of its faces into parts (e.g. by Mesh::splitFace or Mesh::splitEdge):
    /// the leaf of each split face gets a subtree with its old triangles and the new parts;
    /// this is a faster alternative to full tree rebuild (but the tree after the update might be less efficient)
    /// \param mesh same mesh for which this tree was constructed, but after the splits
    /// \param new2Old maps every new face (not present in the tree) to the face of this tree it was split from;
    /// it must contain only the splits made since the tree was built or last updated
    MRMESH_API void addSplitFaces( const Mesh & mesh, const FaceHashMap & new2Old );

private:
    AABBTree( const AABBTree & ) = default;
    AABBTree & operator =( const AABBTree & ) = default;
    friend class UniqueThreadSafeOwner<AABBTree>;
    friend class SharedThreadSafeOwner<AABBTree>;

    Vector<SimdNode> simdNodes_; ///< the children follow their parent
    Vector<SimdLeaf> simdLeaves_;
    Vector<LeafTri> leafTris_;
    int numTris_ = 0;

    /// sets triangle slot i of leaf l with the coordinates of its vertices
    void setLeafTri_( int l, int i, const LeafTri & lt, const VertCoords & points );

    /// sets the coordinates of the vertices of the triangle in slot i of leaf l
    void setLeafTriGeometry_( int l, int i, const ThreeVertIds & verts, const VertCoords & points );
};

/// \}

} // namespace MR
