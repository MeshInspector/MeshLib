#pragma once

#include "MRphmap.h"

namespace MR
{

// mapping among elements of source mesh, from which a part is taken, and target (this) mesh
struct PartMapping
{
    // from.id -> this.id
    // hash maps are used to minimize memory consumption when only a small portion of source mesh is copied
    FaceHashMap * src2tgtFaces = nullptr;
    VertHashMap * src2tgtVerts = nullptr;
    WholeEdgeHashMap * src2tgtEdges = nullptr;
    // this.id -> from.id
    FaceMap * tgt2srcFaces = nullptr;
    VertMap * tgt2srcVerts = nullptr;
    WholeEdgeMap * tgt2srcEdges = nullptr;
};

// the class to convert mappings from new HashMap format to old Vector format
class HashToVectorMappingConverter
{
public:
    MRMESH_API HashToVectorMappingConverter( const MeshTopology & srcTopology, FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap );
    const PartMapping & getPartMapping() const { return map_; }
    MRMESH_API ~HashToVectorMappingConverter(); //conversion takes place here

private:
    FaceMap * outFmap_ = nullptr;
    VertMap * outVmap_ = nullptr;
    WholeEdgeMap * outEmap_ = nullptr;
    PartMapping map_;
    FaceHashMap src2tgtFaces_;
    VertHashMap src2tgtVerts_;
    WholeEdgeHashMap src2tgtEdges_;
};

}
