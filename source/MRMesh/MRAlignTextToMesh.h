#pragma once

#ifndef MRMESH_NO_LABEL
#include "MRMeshFwd.h"
#include <tl/expected.hpp>
#include "MRId.h"
#include "MRVector3.h"
#include "MRSymbolMesh.h"

namespace MR
{
// This structure represents point on mesh, by EdgeId (point should be in left triangle of this edge) and coordinate
struct EdgeIdAndCoord
{
    EdgeId id;
    Vector3f coord;
};

struct TextMeshAlignParams : SymbolMeshParams
{
    // Start coordinate on mesh, represent lowest left corner of text
    EdgeIdAndCoord startPoint;
    // Position of the startPoint in a text bounding box
    // (x, y), where 0 <= x,y <= 1,  x - horizontal, y - vertical
    // (0, 0) - bottom left, (0, 1) - bottom right, (0.5, 0.5) - center, (1, 1) - top right
    Vector2f pivotPoint{0.0f, 0.0f};
    // Direction of text
    Vector3f direction;
    // Font height, meters
    float fontHeight{1.0f};
    // Text mesh inside and outside offset of input mesh
    float surfaceOffset{1.0f};
    // Maximum possible movement of text mesh alignment, meters
    float textMaximumMovement{2.5f};
};

MRMESH_API tl::expected<Mesh, std::string> alignTextToMesh( const Mesh& mesh, const AffineXf3f& xf, const TextMeshAlignParams& params );
}
#endif
