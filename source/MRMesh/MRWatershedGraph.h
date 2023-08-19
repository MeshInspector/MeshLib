#pragma once

#include "MRGraph.h"
#include "MRUnionFind.h"
#include <cfloat>

namespace MR
{

/// graphs representing rain basins on the mesh
class WatershedGraph
{
public:
    /// constructs the graph from given mesh topology, heights in vertices, and initial subdivision on basins
    MRMESH_API void construct( const MeshTopology & topology, const VertScalars & heights, const Vector<int, FaceId> & face2basin, int numBasins );

    /// returns underlying graph where each basin is a vertex
    [[nodiscard]] const Graph & graph() const { return graph_; }

    /// finds the lowest boundary between basins and its height, which is defined
    /// as the minimal different between lowest boundary point and lowest point in a basin
    [[nodiscard]] MRMESH_API std::pair<Graph::EdgeId, float> findLowestBd() const;

    /// merge two basins sharing given boundary
    MRMESH_API void mergeViaBd( Graph::EdgeId bd );

    /// returns the mesh edges between current basins
    [[nodiscard]] MRMESH_API UndirectedEdgeBitSet getBasinEdges( const MeshTopology & topology, const Vector<int, FaceId> & face2basin ) const;

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

    mutable UnionFind<Graph::VertId> ufBasins_;
};

} //namespace MR
