#pragma once
#include <cassert>
#include <vector>

namespace MR
{

/// one edge to add, connecting the origins of the two edges given by the codes;
/// a not-negative code is an EdgeId, a negative one denotes an edge that an earlier item
/// of the same plan creates - see \ref FillHoleItemEdge
struct FillHoleItem
{
    int edgeCode1, edgeCode2;
};

/// an edge created by an item of the plan, given in the same way as EdgeId gives an edge of a mesh:
/// what the edge is, plus which of its two directions is meant
struct FillHoleItemEdge
{
    /// index of the item creating the edge
    int item = 0;
    /// take that edge in the opposite direction, so with the other origin
    bool sym = false;

    /// encodes this edge in a negative code
    [[nodiscard]] int encode() const
    {
        assert( item >= 0 );
        return -( 2 * item + int( sym ) + 1 );
    }

    /// decodes a negative code back in the edge it denotes
    [[nodiscard]] static FillHoleItemEdge decode( int code )
    {
        assert( code < 0 );
        const int c = -( code + 1 );
        return { .item = c >> 1, .sym = ( c & 1 ) != 0 };
    }
};

/// concise representation of proposed hole triangulation
struct HoleFillPlan
{
    std::vector<FillHoleItem> items;
    /// the number of triangles in the filling;
    /// zero means that the plan only adds the edges of items and creates no faces,
    /// e.g. it splits a hole in several holes
    int numTris = 0;
};

}
