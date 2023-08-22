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
    // associated with each vertex in graph
    struct BasinInfo
    {
        VertId lowestVert; ///< in the whole basin
    };

    // associated with each edge in graph
    struct BdInfo
    {
        VertId lowestVert; ///< on this boundary
    };

public:
    /// constructs the graph from given mesh topology, heights in vertices, and initial subdivision on basins
    MRMESH_API WatershedGraph( const MeshTopology & topology, const VertScalars & heights, const Vector<int, FaceId> & face2basin, int numBasins );

    /// returns height at given vertex or FLT_MAX if the vertex is invalid
    [[nodiscard]] float getHeightAt( VertId v ) const { return getAt( heights_, v, FLT_MAX ); }

    /// returns underlying graph where each basin is a vertex
    [[nodiscard]] const Graph & graph() const { return graph_; }
    
    /// returns data associated with given basin
    [[nodiscard]] const BasinInfo & basinInfo( Graph::VertId v ) const { return basins_[v]; }

    /// returns data associated with given boundary between basins
    [[nodiscard]] const BdInfo & bdInfo( Graph::EdgeId e ) const { return bds_[e]; }

    /// finds the lowest boundary between basins and its height, which is defined
    /// as the minimal different between lowest boundary point and lowest point in a basin
    [[nodiscard]] MRMESH_API std::pair<Graph::EdgeId, float> findLowestBd() const;

    /// merge two basins sharing given boundary
    MRMESH_API void mergeViaBd( Graph::EdgeId bd );

    /// returns the mesh faces of given basin
    [[nodiscard]] MRMESH_API FaceBitSet getBasinFaces( Graph::VertId basin ) const;

    /// returns the mesh faces of given basin with at least one vertex below given level
    [[nodiscard]] MRMESH_API FaceBitSet getBasinFacesBelowLevel( Graph::VertId basin, float waterLevel ) const;

    /// returns the mesh edges between current basins
    [[nodiscard]] MRMESH_API UndirectedEdgeBitSet getInterBasinEdges() const;

private:
    const MeshTopology & topology_;
    const VertScalars & heights_;
    const Vector<int, FaceId> & face2iniBasin_;

    Graph graph_;
    Vector<BasinInfo, Graph::VertId> basins_;
    Vector<BdInfo, Graph::EdgeId> bds_;
    mutable UnionFind<Graph::VertId> ufBasins_;
};

} //namespace MR
