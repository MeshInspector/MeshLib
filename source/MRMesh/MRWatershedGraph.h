#pragma once

#include "MRGraph.h"
#include <cfloat>

namespace MR
{

/// graphs representing rain basins on the mesh
class WatershedGraph
{
public:
    MRMESH_API void construct( const MeshTopology & topology, const VertScalars & heights, const Vector<int, FaceId> & face2basin, int numBasins );

private:
    Graph graph_;

    // associated with each vertex in graph
    struct BasinInfo
    {
        float lowestHeight = FLT_MAX;
    };
    Vector<BasinInfo, Graph::VertId> basins_;

    // associated with each edge in graph
    struct BdInfo
    {
        float lowestHeight = FLT_MAX;
    };
    Vector<BdInfo, Graph::EdgeId> bds_;
};

} //namespace MR
