#pragma once

#include "MRMeshFwd.h"
#include <functional>

namespace MR
{

/// \defgroup PolylineSubdivideGroup Polyline Subdivide
/// \ingroup PolylineAlgorithmGroup
/// \{

struct PolylineSubdivideSettings
{
    /// Subdivision is stopped when all edges are not longer than this value
    float maxEdgeLen = 0;
    /// Maximum number of edge splits allowed
    int maxEdgeSplits = 1000;
    /// Region on polyline to be subdivided: both edge vertices must be there to allow spitting,
    /// it is updated during the operation
    VertBitSet * region = nullptr;
    /// New vertices appeared during subdivision will be added here
    VertBitSet * newVerts = nullptr;
    /// this function is called each time a new vertex has been created
    std::function<void(VertId)> onVertCreated;
};

/// Split edges in polyline according to the settings;\n
/// \return The total number of edge splits performed
MRMESH_API int subdividePolyline( Polyline2 & polyline, const PolylineSubdivideSettings & settings = {} );
MRMESH_API int subdividePolyline( Polyline3 & polyline, const PolylineSubdivideSettings & settings = {} );

/// \}

}
