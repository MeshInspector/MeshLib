#pragma once

#include "MRMeshBuilderTypes.h"
#include "MRVector.h"
#include "MRVector3.h"

namespace MR
{

/// polygonal mesh as it is stored in many file formats: vertex coordinates and polygons referencing them,
/// without topology, so polygons are not triangulated and no vertex is duplicated
struct [[nodiscard]] FaceSoup
{
    VertCoords points;

    /// vertex ids of all polygons, one polygon after another
    Vector<VertId, int> verts;

    /// the range in verts of each polygon
    Vector<MeshBuilder::VertSpan, FaceId> faces;
};

} //namespace MR
