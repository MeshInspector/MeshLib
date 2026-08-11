#pragma once
#include <cassert>
#include <vector>

namespace MR
{

/// one edge to add, connecting the origins of the two edges given by the codes;
/// a not-negative code is an EdgeId, a negative one refers to the edge created by an earlier item
/// of the same plan - see \ref encodeFillHoleItemRef
struct FillHoleItem
{
    int edgeCode1, edgeCode2;
};

/// reference to the edge created by an earlier item of the same plan
struct FillHoleItemRef
{
    /// index of the item creating the edge
    int item = 0;
    /// take that edge in the opposite direction, so with the other origin
    bool sym = false;
};

/// encodes the reference to the edge of an earlier item in a negative code
[[nodiscard]] inline int encodeFillHoleItemRef( FillHoleItemRef ref )
{
    assert( ref.item >= 0 );
    return -( 2 * ref.item + int( ref.sym ) + 1 );
}

/// decodes a negative code back in the reference it holds
[[nodiscard]] inline FillHoleItemRef decodeFillHoleItemRef( int code )
{
    assert( code < 0 );
    const int r = -( code + 1 );
    return { .item = r >> 1, .sym = ( r & 1 ) != 0 };
}

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
