#pragma once
#include <vector>

struct FillHoleItem
{
    // if not-negative number then it is edgeid;
    // otherwise it refers to the edge created recently
    int edgeCode1, edgeCode2;
};

/// concise representation of proposed hole triangulation
struct HoleFillPlan
{
    std::vector<FillHoleItem> items;
    int numTris = 0; // the number of triangles in the filling
};