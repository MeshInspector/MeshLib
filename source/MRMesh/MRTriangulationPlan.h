#pragma once
#include "MRPch/MRBindingMacros.h"
#include <vector>

namespace MR
{

struct TriangulationPlanItem
{
    // if not-negative number then it is edgeid;
    // otherwise it refers to the edge created recently
    int edgeCode1, edgeCode2;
};

/// concise representation of the edges to add in a topology, e.g. of a proposed hole triangulation
struct TriangulationPlan
{
    std::vector<TriangulationPlanItem> items;
    int numTris = 0; // the number of triangles in the filling
};

using FillHoleItem [[deprecated( "Use `TriangulationPlanItem` instead." )]] MR_BIND_IGNORE = TriangulationPlanItem;
using HoleFillPlan [[deprecated( "Use `TriangulationPlan` instead." )]] MR_BIND_IGNORE = TriangulationPlan;

}
