#pragma once

#include "MRBitSet.h"

namespace MR::detail
{

/// TriMesh with a region of triangles: only the triangles of the region and their vertices are valid
struct TriMeshPart
{
    const Triangulation & tris;
    const VertCoords & points;
    const FaceBitSet & region;
    VertBitSet regionVerts; ///< vertices of region triangles

    /// region must not have bits beyond m.tris
    MRMESH_API TriMeshPart( const TriMesh & m, const FaceBitSet & r );
};

} //namespace MR::detail
