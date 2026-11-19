#pragma once

#include "MRBox.h"
#include "MRPch/MRBindingMacros.h"
#include <limits>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

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

/// internal node of AABBTree or AABBTreePoints: the boxes of its (up to 4) children in SoA order and the references to them;
/// lets a traversal test all children of a node at once
struct MR_BIND_IGNORE AABBTreeSimdNode : AABBTreeSimdBoxes
{
    /// i-th child: the index in simdNodes() if > 0, otherwise ~(the index of the leaf); 0 in unused slots (the root is never a child)
    int child[4];

    AABBTreeSimdNode() = default;
    explicit AABBTreeSimdNode( NoInit ) noexcept : AABBTreeSimdBoxes( noInit ) {}
};

/// \}

} // namespace MR
