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
