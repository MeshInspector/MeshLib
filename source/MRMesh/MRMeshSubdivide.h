#pragma once

#include "MRMeshFwd.h"

namespace MR
{

struct SubdivideSettings
{
    /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
    float maxEdgeLen = 0;
    /// Maximum number of edge splits allowed
    int maxEdgeSplits = 1000;
    /// Improves local mesh triangulation after each edge flip if it does not make too big surface deviation
    float maxDeviationAfterFlip = 1;
    /// Region on mesh to be subdivided, it is updated during the operation
    FaceBitSet * region = nullptr;
    /// New vertices appeared during subdivision will be added here
    VertBitSet * newVerts = nullptr;
    /// If false do not touch border edges (cannot subdivide lone faces)\n
    /// use \ref MR::findRegionOuterFaces to find boundary faces
    bool subdivideBorder = true;
    /// If subdivideBorder is off subdivider can produce narrow triangles near border\n
    /// this parameter prevents subdivision of such triangles
    float critAspectRatio = 20.0f;
    /// this function is called each time a new vertex has been created, but before the ring is made Delone
    std::function<void(VertId)> onVertCreated;
};

/// Split edges in mesh region according to the settings;\n
/// \returns The total number of edge flips performed
MRMESH_API int subdivideMesh( Mesh & mesh, const SubdivideSettings & settings = {} );

}
