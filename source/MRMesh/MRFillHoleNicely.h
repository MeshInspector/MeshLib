#pragma once

#include "MRMeshFwd.h"

namespace MR
{

struct FillHoleNicelySettings
{
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
    const FillHoleNicelySettings & settings = {} );

} //namespace MR
