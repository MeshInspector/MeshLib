#pragma once

#include "MRMapOrHashMap.h"

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

/// use this adapter to call functions expecting PartMapping parameter to receive src2tgt dense maps
class Src2TgtMaps
{
public:
    MRMESH_API Src2TgtMaps( FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap );
    [[deprecated]] Src2TgtMaps( const MeshTopology &, FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap )
        : Src2TgtMaps( outFmap, outVmap, outEmap ) {}
    MRMESH_API ~Src2TgtMaps(); // maps are moved back to user here

    operator const PartMapping &() const { return map_; }
    const PartMapping & getPartMapping() const { return map_; }

private:
    FaceMap * outFmap_ = nullptr;
    VertMap * outVmap_ = nullptr;
    WholeEdgeMap * outEmap_ = nullptr;
    PartMapping map_;
    FaceMapOrHashMap src2tgtFaces_;
    VertMapOrHashMap src2tgtVerts_;
    WholeEdgeMapOrHashMap src2tgtEdges_;
};

using HashToVectorMappingConverter [[deprecated]] = Src2TgtMaps;

/// use this adapter to call functions expecting PartMapping parameter to receive tgt2src dense maps
class Tgt2SrcMaps
{
public:
    MRMESH_API Tgt2SrcMaps( FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap );
    MRMESH_API ~Tgt2SrcMaps(); // maps are moved back to user here

    operator const PartMapping &() const { return map_; }
    const PartMapping & getPartMapping() const { return map_; }

private:
    FaceMap * outFmap_ = nullptr;
    VertMap * outVmap_ = nullptr;
    WholeEdgeMap * outEmap_ = nullptr;
    PartMapping map_;
    FaceMapOrHashMap tgt2srcFaces_;
    VertMapOrHashMap tgt2srcVerts_;
    WholeEdgeMapOrHashMap tgt2srcEdges_;
};

} //namespace MR
