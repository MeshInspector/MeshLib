#include "MRPartMapping.h"
#include "MRMeshTopology.h"

namespace MR
{

void PartMapping::clear()
{
    if ( src2tgtFaces )
        src2tgtFaces->clear();
    if ( src2tgtVerts )
        src2tgtVerts->clear();
    if ( src2tgtEdges )
        src2tgtEdges->clear();
    if ( tgt2srcFaces )
        tgt2srcFaces->clear();
    if ( tgt2srcVerts )
        tgt2srcVerts->clear();
    if ( tgt2srcEdges )
        tgt2srcEdges->clear();
}

Src2TgtMaps::Src2TgtMaps( FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap )
    : outFmap_( outFmap ), outVmap_( outVmap ), outEmap_( outEmap )
{
    if ( outFmap )
    {
        src2tgtFaces_.setMap( std::move( *outFmap_ ) );
        map_.src2tgtFaces = &src2tgtFaces_;
    }
    if ( outVmap )
    {
        src2tgtVerts_.setMap( std::move( *outVmap_ ) );
        map_.src2tgtVerts = &src2tgtVerts_;
    }
    if ( outEmap )
    {
        src2tgtEdges_.setMap( std::move( *outEmap_ ) );
        map_.src2tgtEdges = &src2tgtEdges_;
    }
}

Src2TgtMaps::~Src2TgtMaps()
{
    if ( outFmap_ )
        *outFmap_ = std::move( *src2tgtFaces_.getMap() );
    if ( outVmap_ )
        *outVmap_ = std::move( *src2tgtVerts_.getMap() );
    if ( outEmap_ )
        *outEmap_ = std::move( *src2tgtEdges_.getMap() );
}

} //namespace MR
