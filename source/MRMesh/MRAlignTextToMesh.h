#pragma once

#ifndef MRMESH_NO_LABEL
#include "MRMeshFwd.h"
#include <tl/expected.hpp>
#include "MRId.h"
#include "MRVector3.h"
#include "MRSymbolMesh.h"
#include "MRMeshTriPoint.h"

namespace MR
{

struct TextMeshAlignParams : SymbolMeshParams
{
    // Start coordinate on mesh, represent lowest left corner of text
    MeshTriPoint startPoint;
    // Direction of text
    Vector3f direction;
    // Text normal to surface, if nullptr - use mesh normal at `startPoint`
    const Vector3f* textNormal{nullptr};
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
