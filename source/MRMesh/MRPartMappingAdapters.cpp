#include "MRPartMappingAdapters.h"
#include "MRId.h"

namespace MR
{

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

Tgt2SrcMaps::Tgt2SrcMaps( FaceMap * outFmap, VertMap * outVmap, WholeEdgeMap * outEmap )
    : outFmap_( outFmap ), outVmap_( outVmap ), outEmap_( outEmap )
{
    if ( outFmap )
    {
        tgt2srcFaces_.setMap( std::move( *outFmap_ ) );
        map_.tgt2srcFaces = &tgt2srcFaces_;
    }
    if ( outVmap )
    {
        tgt2srcVerts_.setMap( std::move( *outVmap_ ) );
        map_.tgt2srcVerts = &tgt2srcVerts_;
    }
    if ( outEmap )
    {
        tgt2srcEdges_.setMap( std::move( *outEmap_ ) );
        map_.tgt2srcEdges = &tgt2srcEdges_;
    }
}

Tgt2SrcMaps::~Tgt2SrcMaps()
{
    if ( outFmap_ )
        *outFmap_ = std::move( *tgt2srcFaces_.getMap() );
    if ( outVmap_ )
        *outVmap_ = std::move( *tgt2srcVerts_.getMap() );
    if ( outEmap_ )
        *outEmap_ = std::move( *tgt2srcEdges_.getMap() );
}

} //namespace MR
