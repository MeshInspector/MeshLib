#pragma once

#include "MRMeshTriPoint.h"
#include "MRVector.h"

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
        Polyline3 * outPolyline = nullptr, UndirectedEdgeScalars * outFlowPerEdge = nullptr );

private:
    const Mesh & mesh_;
    const VertScalars & heights_;
    VertMap downFlowVert_; // for each vertex stores what next vertex is on flow path (invalid vertex for local minima)
    Vector<SurfacePath, VertId> downPath_; // till next vertex
    std::vector<VertId> vertsSortedDesc_; // all vertices sorted in descending heights order
};

} //namespace MR
