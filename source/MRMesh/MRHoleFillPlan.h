#pragma once
#include <vector>

namespace MR
{

/// one edge to add, connecting the origins of the two edges given by the codes
struct FillHoleItem
{
    /// if not-negative number then it is edgeid;
    /// otherwise it refers to the edge created by an earlier item: -( 2 * item + sym + 1 ),
    /// where sym tells to take that edge in the opposite direction (so with the other origin)
    int edgeCode1, edgeCode2;
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
