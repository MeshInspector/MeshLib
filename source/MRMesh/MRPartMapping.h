#pragma once

#include "MRphmap.h"

namespace MR
{

/// mapping among elements of source mesh, from which a part is taken, and target mesh
struct PartMapping
{
    // source.id -> target.id
    // hash maps minimize memory consumption when only a small portion of source mesh is copied
    FaceHashMap * src2tgtFaceHashMap = nullptr;
    VertHashMap * src2tgtVertHashMap = nullptr;
    WholeEdgeHashMap * src2tgtWholeEdgeHashMap = nullptr;

    // target.id -> source.id
    // dense vectors are better by speed and memory when target mesh was empty before copying
    FaceMap * tgt2srcFaceMap = nullptr;
    VertMap * tgt2srcVertMap = nullptr;
    WholeEdgeMap * tgt2srcWholeEdgeMap = nullptr;

    /// clears all memeber maps
    MRMESH_API void clear();
};

/// the class to convert mappings from new HashMap format to old Vector format
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
