#pragma once

#include "MRMeshTriPoint.h"
#include "MRVector.h"
#include "MRPolyline.h"
#include <functional>

namespace MR
{

struct FlowOrigin
{
    /// point on the mesh, where this flow starts
    MeshTriPoint point;
    /// amount of flow, e.g. can be proportional to the horizontal area associated with the start point
    float amount = 1;
};

/// this class can track multiple flows and find in each mesh vertex the amount of water reached it
class FlowAggregator
{
public:
    /// prepares the processing of given mesh with given height in each vertex
    MRMESH_API FlowAggregator( const Mesh & mesh, const VertScalars & heights );
    /// tracks multiple flows
    /// \param starts the origin of each flow (should be uniformly sampled over the terrain)
    /// \param outPolyline output lines of all flows
    /// \param outFlowPerEdge flow in each line of outPolyline
    /// \return the flow reached each mesh vertex
    MRMESH_API VertScalars computeFlow( const std::vector<FlowOrigin> & starts,
        Polyline3 * outPolyline = nullptr, UndirectedEdgeScalars * outFlowPerEdge = nullptr ) const;
    // same with all amounts equal to 1
    MRMESH_API VertScalars computeFlow( const std::vector<MeshTriPoint> & starts,
        Polyline3 * outPolyline = nullptr, UndirectedEdgeScalars * outFlowPerEdge = nullptr ) const;
    // general version that supplies starts in a functional way
    MRMESH_API VertScalars computeFlow( size_t numStarts,
        const std::function<MeshTriPoint(size_t)> & startById, ///< can return invalid point that will be ignored
        const std::function<float(size_t)> & amountById,
        Polyline3 * outPolyline = nullptr, UndirectedEdgeScalars * outFlowPerEdge = nullptr ) const;

    struct Flows
    {
        Polyline3 polyline;
        UndirectedEdgeScalars flowPerEdge;
    };

    /// tracks multiple flows
    /// \param starts the origin of each flow (should be uniformly sampled over the terrain)
    /// \return the flows grouped by the final destination vertex
    MRMESH_API HashMap<VertId, Flows> computeFlowsPerBasin( const std::vector<FlowOrigin> & starts ) const;
    // same with all amounts equal to 1
    MRMESH_API HashMap<VertId, Flows> computeFlowsPerBasin( const std::vector<MeshTriPoint> & starts ) const;
    // general version that supplies starts in a functional way
    MRMESH_API HashMap<VertId, Flows> computeFlowsPerBasin( size_t numStarts,
        const std::function<MeshTriPoint(size_t)> & startById, ///< can return invalid point that will be ignored
        const std::function<float(size_t)> & amountById ) const;

    /// finds the edges on the mesh that divides catchment basin
    /// (every triangle is attributed to the final destination point based on the path originated from its centroid)
    [[nodiscard]] MRMESH_API UndirectedEdgeBitSet computeCatchmentDelineation() const;

private:
    const Mesh & mesh_;
    const VertScalars & heights_;
    VertMap downFlowVert_; // for each vertex stores what next vertex is on flow path (invalid vertex for local minima)
    VertMap rootVert_;     // for each vertex stores the final vertex is on flow path (self-vertex for local minima)
    Vector<SurfacePath, VertId> downPath_; // till next vertex
    std::vector<VertId> vertsSortedDesc_; // all vertices sorted in descending heights order
};

} //namespace MR
