#include "MRPartMapping.h"
#include "MRMeshTopology.h"

namespace MR
{

HashToVectorMappingConverter::HashToVectorMappingConverter( const MeshTopology & srcTopology, FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap )
    : outFmap_( outFmap ), outVmap_( outVmap ), outEmap_( outEmap )
{
    if ( outFmap )
    {
        map_.src2tgtFaces = &src2tgtFaces_;
        outFmap->clear();
        outFmap->resize( (int)srcTopology.lastValidFace() + 1 );
    }
    if ( outVmap )
    {
        map_.src2tgtVerts = &src2tgtVerts_;
        outVmap->clear();
        outVmap->resize( (int)srcTopology.lastValidVert() + 1 );
    }
    if ( outEmap )
    {
        map_.src2tgtEdges = &src2tgtEdges_;
        outEmap->clear();
        outEmap->resize( srcTopology.undirectedEdgeSize() );
    }
}

HashToVectorMappingConverter::~HashToVectorMappingConverter()
{
    if ( outFmap_ )
        *outFmap_ = std::move( *src2tgtFaces_.getMap() );
    if ( outVmap_ )
        *outVmap_ = std::move( *src2tgtVerts_.getMap() );
    if ( outEmap_ )
        *outEmap_ = std::move( *src2tgtEdges_.getMap() );
}

} //namespace MR
