#pragma once

#include "MRMeshFwd.h"
#include "MRMeshFillHole.h"

namespace MR
{

struct FillHoleNicelySettings
{
    /// how to triangulate the hole, must be specified by the user
    FillHoleParams triangulateParams;

    /// If false then additional vertices are created inside the patch for best mesh quality
    bool triangulateOnly = false;

    /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
    float maxEdgeLen = 0;

    /// Maximum number of edge splits allowed
    int maxEdgeSplits = 1000;

    /// Whether to make patch over the hole smooth both inside and on its boundary with existed surface
    bool smoothCurvature = true;

    /// optional uv-coordinates of vertices; if provided then elements corresponding to new vertices will be added there
    VertUVCoords * uvCoords = {};

    /// optional colors of vertices; if provided then elements corresponding to new vertices will be added there
    VertColors * colorMap = {};
};

/// fills a hole in mesh specified by one of its edge,
/// optionally subdivides new patch on smaller triangles,
/// optionally make smooth connection with existing triangles outside the hole
/// \return triangles of the patch
MRMESH_API FaceBitSet fillHoleNicely( Mesh & mesh,
    EdgeId holeEdge, ///< left of this edge must not have a face and it will be filled
    const FillHoleNicelySettings & settings );

} //namespace MR
