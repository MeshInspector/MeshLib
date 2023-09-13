#pragma once

#include "MRGraph.h"
#include <cassert>
#include <cfloat>

namespace MR
{

/// graphs representing rain basins on the mesh
class WatershedGraph
{
public:
    /// associated with each vertex in graph
    struct BasinInfo
    {
        VertId lowestVert; ///< in the whole basin
        float lowestLevel = FLT_MAX; ///< lowest level (z-coordinate of lowestVert) in the basin
        float area = 0;    ///< precipitation area that flows in this basin
        float lowestBdLevel = FLT_MAX; ///< lowest position on the boundary of the basin
        float fullVolume = 0; ///< full water volume to be accumulated in the basin till water reaches the lowest height on the boundary
        float remVolume = 0;  ///< remaining water volume to be accumulated in the basin till water reaches the lowest height on the boundary
        float lastUpdateTime = 0; ///< the time when remVolume was last updated
        Graph::VertId overflowTo; ///< when level=lowestBdLevel, volume=0, all water from this basin overflows to given basin, and this.area becomes equal to 0

        float timeTillOverflow() const
        { 
            assert( !overflowTo );
            return remVolume / area;
        }

        /// approximate current level of water (z-coordinate) in the basin
        float approxLevel() const
        { 
            const auto p = remVolume / fullVolume;
            assert( p >= 0 && p <= 1 );
            return p * lowestLevel + ( 1 - p ) * lowestBdLevel;
        }

        void update( float time )
        {
            assert( time >= lastUpdateTime );
            remVolume -= ( time - lastUpdateTime ) * area;
            if ( remVolume < 0 ) // due to rounding errors
                remVolume = 0;
            lastUpdateTime = time;
        }
    };

    /// associated with each edge in graph
    struct BdInfo
    {
        VertId lowestVert; ///< on this boundary
    };

public:
    /// constructs the graph from given mesh, heights in z-coordinate, and initial subdivision on basins
    MRMESH_API WatershedGraph( const Mesh & mesh, const Vector<int, FaceId> & face2basin, int numBasins );

    /// returns height at given vertex or FLT_MAX if the vertex is invalid
    [[nodiscard]] MRMESH_API float getHeightAt( VertId v ) const;

    /// returns underlying graph where each basin is a vertex
    [[nodiscard]] const Graph & graph() const { return graph_; }
    
    /// returns the current number of basins (excluding special "outside" basin)
    [[nodiscard]] int numBasins() const { return (int)graph_.validVerts().count() - 1; }

    /// returns data associated with given basin
    [[nodiscard]] const BasinInfo & basinInfo( Graph::VertId v ) const { return basins_[v]; }
    [[nodiscard]] BasinInfo & basinInfo( Graph::VertId v ) { return basins_[v]; }

    /// returns data associated with given boundary between basins
    [[nodiscard]] const BdInfo & bdInfo( Graph::EdgeId e ) const { return bds_[e]; }
    [[nodiscard]] BdInfo & bdInfo( Graph::EdgeId e ) { return bds_[e]; }

    /// returns special "basin" representing outside areas of the mesh
    [[nodiscard]] Graph::VertId outsideId() const { return outsideId_; }

    /// for valid basin returns self id; for invalid basin returns the id of basin it was merged in
    [[nodiscard]] MRMESH_API Graph::VertId getRootBasin( Graph::VertId v ) const;

    /// returns basin where the flow from this basin goes (it can be self id if the basin is not full yet)
    [[nodiscard]] MRMESH_API Graph::VertId flowsTo( Graph::VertId v ) const;

    /// replaces parent of each basin with its computed root;
    /// this speeds up following calls to getRootBasin()
    MRMESH_API void setParentsToRoots();

    /// finds the lowest boundary between basins and its height, which is defined
    /// as the minimal different between lowest boundary point and lowest point in a basin
    [[nodiscard]] MRMESH_API std::pair<Graph::EdgeId, float> findLowestBd() const;

    /// merges basin v1 into basin v0, v1 is deleted after that, returns v0
    MRMESH_API Graph::VertId merge( Graph::VertId v0, Graph::VertId v1 );

    /// merges two basins sharing given boundary, returns remaining basin
    MRMESH_API Graph::VertId mergeViaBd( Graph::EdgeId bd );

    /// returns the mesh faces of given basin
    [[nodiscard]] MRMESH_API FaceBitSet getBasinFaces( Graph::VertId basin ) const;

    /// returns the mesh faces of given basin with at least one vertex below given level
    [[nodiscard]] MRMESH_API FaceBitSet getBasinFacesBelowLevel( Graph::VertId basin, float waterLevel ) const;

    /// returns water volume in basin when its surface reaches given level, which must be in between
    /// the lowest basin level and the lowest level on basin's boundary
    [[nodiscard]] MRMESH_API double computeBasinVolume( Graph::VertId basin, float waterLevel ) const;

    /// returns the mesh edges between current basins
    [[nodiscard]] MRMESH_API UndirectedEdgeBitSet getInterBasinEdges() const;

private:
    const Mesh & mesh_;
    const Vector<int, FaceId> & face2iniBasin_;

    Graph graph_;
    Vector<BasinInfo, Graph::VertId> basins_;
    Vector<BdInfo, Graph::EdgeId> bds_;

    /// special "basin" representing outside areas of the mesh
    Graph::VertId outsideId_;

    /// for valid basin, parent is the same; for invalid basin, sequence of parents point on valid root basin
    Vector<Graph::VertId, Graph::VertId> parentBasin_;
};

} //namespace MR
