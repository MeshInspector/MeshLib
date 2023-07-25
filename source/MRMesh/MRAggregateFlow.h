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

class FlowAggregator
{
public:
    MRMESH_API FlowAggregator( const Mesh & mesh, const VertScalars & field );
    MRMESH_API VertScalars computeFlow( const std::vector<FlowOrigin> & starts,
        Polyline3 * outPolyline = nullptr, UndirectedEdgeScalars * outFlowPerEdge = nullptr );

private:
    const Mesh & mesh_;
    const VertScalars & field_;
    VertMap downFlowVert_; // for each vertex stores what next vertex is on flow path (invalid vertex for local minima)
    Vector<SurfacePath, VertId> downPath_; // till next vertex
    std::vector<VertId> vertsSortedDesc_; // all vertices sorted in descending field order
};

} //namespace MR
