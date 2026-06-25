#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// mapping among elements of source mesh, from which a part is taken, and target mesh
struct PartMapping
{
    // source.id -> target.id
    // each map here can be either dense vector or hash map, the type is set by the user and preserved by mesh copying functions;
    // dense maps are better by speed and memory when source mesh is packed and must be copied entirely;
    // hash maps minimize memory consumption when only a small portion of source mesh is copied
    FaceMapOrHashMap * src2tgtFaces = nullptr;
    VertMapOrHashMap * src2tgtVerts = nullptr;
    WholeEdgeMapOrHashMap * src2tgtEdges = nullptr;

    // target.id -> source.id
    // dense vectors are better by speed and memory when target mesh was empty before copying
    FaceMapOrHashMap * tgt2srcFaces = nullptr;
    VertMapOrHashMap * tgt2srcVerts = nullptr;
    WholeEdgeMapOrHashMap * tgt2srcEdges = nullptr;

    /// clears all member maps
    MRMESH_API void clear();
};

} //namespace MR
