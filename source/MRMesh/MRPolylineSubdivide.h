#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
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
    /// This option works best for natural lines, where all segments have similar size,
    /// and no sharp angles in between
    bool useCurvature = false;
    /// this function is called each time a new vertex has been created
    std::function<void(VertId)> onVertCreated;
    /// callback to report algorithm progress and cancel it by user request
    ProgressCallback progressCallback = {};
};

/// Split edges in polyline according to the settings;\n
/// \return The total number of edge splits performed
MRMESH_API int subdividePolyline( Polyline2 & polyline, const PolylineSubdivideSettings & settings = {} );
MRMESH_API int subdividePolyline( Polyline3 & polyline, const PolylineSubdivideSettings & settings = {} );

/// \}

}
