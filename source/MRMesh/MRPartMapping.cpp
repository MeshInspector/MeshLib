#include "MRPartMapping.h"
#include "MRMeshTopology.h"

namespace MR
{

HashToVectorMappingConverter::HashToVectorMappingConverter( const MeshTopology & srcTopology, FaceMap * outFmap, VertMap * outVmap, EdgeMap * outEmap )
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
        outEmap->resize( (int)srcTopology.lastNotLoneEdge() + 1 );
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
