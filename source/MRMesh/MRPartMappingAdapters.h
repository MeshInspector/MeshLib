#pragma once

#include "MRPartMapping.h"
#include "MRMapOrHashMap.h"

namespace MR
{

/// use this adapter to call functions expecting PartMapping parameter to receive src2tgt dense maps
class Src2TgtMaps
{
public:
    MRMESH_API Src2TgtMaps( FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap );
    MRMESH_API ~Src2TgtMaps(); // maps are moved back to user here

    Src2TgtMaps( const Src2TgtMaps& ) = delete;
    Src2TgtMaps( Src2TgtMaps&& ) noexcept = default;

    Src2TgtMaps & operator = ( const Src2TgtMaps& ) = delete;
    Src2TgtMaps & operator = ( Src2TgtMaps&& ) noexcept = default;

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

/// use this adapter to call functions expecting PartMapping parameter to receive tgt2src dense maps
class Tgt2SrcMaps
{
public:
    MRMESH_API Tgt2SrcMaps( FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap );
    MRMESH_API ~Tgt2SrcMaps(); // maps are moved back to user here

    Tgt2SrcMaps( const Tgt2SrcMaps& ) = delete;
    Tgt2SrcMaps( Tgt2SrcMaps&& ) noexcept = default;

    Tgt2SrcMaps & operator = ( const Tgt2SrcMaps& ) = delete;
    Tgt2SrcMaps & operator = ( Tgt2SrcMaps&& ) noexcept = default;

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
