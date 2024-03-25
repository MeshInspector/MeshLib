#pragma once

#include "MRVector.h"

namespace MR
{

/// very simple structure for storing mesh of triangles only,
/// without easy navigation between neighbor elements as in Mesh
struct [[nodiscard]] TriMesh
{
    Triangulation tris;
    VertCoords points;
};

} //namespace MR
