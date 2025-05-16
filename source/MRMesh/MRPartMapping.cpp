#include "MRPartMapping.h"
#include "MRMeshTopology.h"

namespace MR
{

void PartMapping::clear()
{
    if ( src2tgtFaceHashMap )
        src2tgtFaceHashMap->clear();
    if ( src2tgtVertHashMap )
        src2tgtVertHashMap->clear();
    if ( src2tgtWholeEdgeHashMap )
        src2tgtWholeEdgeHashMap->clear();
    if ( tgt2srcFaceMap )
        tgt2srcFaceMap->clear();
    if ( tgt2srcVertMap )
        tgt2srcVertMap->clear();
    if ( tgt2srcWholeEdgeMap )
        tgt2srcWholeEdgeMap->clear();
}

HashToVectorMappingConverter::HashToVectorMappingConverter( const MeshTopology & srcTopology, FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap )
    : outFmap_( outFmap ), outVmap_( outVmap ), outEmap_( outEmap )
{
    if ( outFmap )
    {
        map_.src2tgtFaceHashMap = &src2tgtFaces_;
        outFmap->clear();
        outFmap->resize( (int)srcTopology.lastValidFace() + 1 );
    }
    if ( outVmap )
    {
        map_.src2tgtVertHashMap = &src2tgtVerts_;
        outVmap->clear();
        outVmap->resize( (int)srcTopology.lastValidVert() + 1 );
    }
    if ( outEmap )
    {
        map_.src2tgtWholeEdgeHashMap = &src2tgtEdges_;
        outEmap->clear();
        outEmap->resize( srcTopology.undirectedEdgeSize() );
    }
}

HashToVectorMappingConverter::~HashToVectorMappingConverter()
{
    if ( outFmap_ )
    {
        for ( const auto & [ fromFace, thisFace ] : src2tgtFaces_ )
            (*outFmap_)[fromFace] = thisFace;
    }
    if ( outVmap_ )
    {
        for ( const auto & [ fromVert, thisVert ] : src2tgtVerts_ )
            (*outVmap_)[fromVert] = thisVert;
    }
    if ( outEmap_ )
    {
        for ( const auto & [ fromEdge, thisEdge ] : src2tgtEdges_ )
            (*outEmap_)[fromEdge] = thisEdge;
    }
}

} //namespace MR
