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
        float area = 0;    ///< precipitation area that flows in this basin (and if it is full, continue flowing next)
        float lowestBdLevel = FLT_MAX; ///< lowest position on the boundary of the basin
        float maxVolume = 0; ///< full water volume to be accumulated in the basin till water reaches the lowest height on the boundary
        float accVolume = 0; ///< accumulated water volume in the basin so far
        float lastUpdateAmount = 0; ///< the amount when accVolume was last updated
        float lastMergeLevel = FLT_MAX; ///< water level in the basin when it was formed (by merge or creation)
        float lastMergeVolume = 0; ///< water volume in the basin when it was formed (by merge or creation)
        Graph::EdgeId overflowVia; ///< when level=lowestBdLevel, volume=0, all water from this basin overflows via this boundary

        BasinInfo() {} // Apparently I need this for `MR::Vector` to register default-constructibility inside the enclosing class.

        /// amount of precipitation (in same units as mesh coordinates and water level),
        /// which can be added before overflowing the basin
        float amountTillOverflow() const
        {
            assert( !overflowVia );
            assert( maxVolume >= accVolume );
            return ( maxVolume - accVolume ) / area;
        }

        /// approximate current level of water (z-coordinate) in the basin
        float approxLevel() const
        {
            assert( lastMergeLevel <= lowestBdLevel );
            assert( lastMergeVolume <= maxVolume );
            if ( maxVolume <= lastMergeVolume )
                return lowestBdLevel;
            const auto p = ( maxVolume - accVolume ) / ( maxVolume - lastMergeVolume );
            assert( p >= 0 && p <= 1 );
            return p * lastMergeLevel + ( 1 - p ) * lowestBdLevel;
        }

        /// updates accumulated volume in the basin to the moment of given precipitation amount
        void updateAccVolume( float amount )
        {
            assert( !overflowVia );
            assert( amount >= lastUpdateAmount );
            accVolume += ( amount - lastUpdateAmount ) * area;
            if ( accVolume > maxVolume ) // due to rounding errors
                accVolume = maxVolume;
            lastUpdateAmount = amount;
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

    /// returns total precipitation area
    [[nodiscard]] float totalArea() const { return totalArea_; }

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

    /// returns the basin where the flow from this basin goes next (it can be self id if the basin is not full yet)
    [[nodiscard]] MRMESH_API Graph::VertId flowsTo( Graph::VertId v ) const;

    /// returns the basin where the flow from this basin finally goes (it can be self id if the basin is not full yet);
    /// \param exceptOutside if true then the method returns the basin that receives water flow from (v) just before outside
    [[nodiscard]] MRMESH_API Graph::VertId flowsFinallyTo( Graph::VertId v, bool exceptOutside = false ) const;

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

    /// returns the mesh faces of each valid basin;
    /// \param joinOverflowBasins if true then overflowing basins will be merged in the target basins (except for overflow in outside)
    [[nodiscard]] MRMESH_API Vector<FaceBitSet, Graph::VertId> getAllBasinFaces( bool joinOverflowBasins = false ) const;

    /// returns the mesh faces of given basin with at least one vertex below given level
    [[nodiscard]] MRMESH_API FaceBitSet getBasinFacesBelowLevel( Graph::VertId basin, float waterLevel ) const;

    /// returns water volume in basin when its surface reaches given level, which must be in between
    /// the lowest basin level and the lowest level on basin's boundary
    [[nodiscard]] MRMESH_API double computeBasinVolume( Graph::VertId basin, float waterLevel ) const;

    /// returns the mesh edges between current basins
    /// \param joinOverflowBasins if true then overflowing basins will be merged in the target basins (except for overflow in outside)
    [[nodiscard]] MRMESH_API UndirectedEdgeBitSet getInterBasinEdges( bool joinOverflowBasins = false ) const;

    /// describes a point where a flow from one basin overflows into another basin
    struct OverflowPoint
    {
        VertId v; // mesh vertex on the boundary of full basin and the other where it overflows
        Graph::VertId fullBasin;
        Graph::VertId overflowTo; // basin where the flow from v goes
    };

    /// returns all overflow points in the graph
    [[nodiscard]] MRMESH_API std::vector<OverflowPoint> getOverflowPoints() const;

    /// computes a map from initial basin id to a valid basin in which it was merged
    /// \param joinOverflowBasins if true then overflowing basins will be merged in the target basins (except for overflow in outside)
    [[nodiscard]] MRMESH_API Vector<Graph::VertId, Graph::VertId> iniBasin2Tgt( bool joinOverflowBasins = false ) const;

private:
    const Mesh & mesh_;
    const Vector<int, FaceId> & face2iniBasin_;

    Graph graph_;
    Vector<BasinInfo, Graph::VertId> basins_;
    Vector<BdInfo, Graph::EdgeId> bds_;
    float totalArea_ = 0;

    /// special "basin" representing outside areas of the mesh
    Graph::VertId outsideId_;

    /// for valid basin, parent is the same; for invalid basin, sequence of parents point on valid root basin
    Vector<Graph::VertId, Graph::VertId> parentBasin_;
};

} //namespace MR
