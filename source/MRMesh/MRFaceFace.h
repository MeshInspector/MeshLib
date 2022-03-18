#pragma once

#include "MRId.h"

namespace MR
{

struct FaceFace
{
    FaceId aFace;
    FaceId bFace;
    FaceFace( FaceId a, FaceId b ) : aFace( a ), bFace( b ) { }
    FaceFace() { };
    bool operator==( const FaceFace& rhs ) const { return ( aFace == rhs.aFace ) && ( bFace == rhs.bFace ); }
};

struct UndirectedEdgeUndirectedEdge
{
    UndirectedEdgeId aUndirEdge;
    UndirectedEdgeId bUndirEdge;
    UndirectedEdgeUndirectedEdge( UndirectedEdgeId a, UndirectedEdgeId b ) : aUndirEdge( a ), bUndirEdge( b )
    {}
    UndirectedEdgeUndirectedEdge()
    {};
    bool operator==( const UndirectedEdgeUndirectedEdge& rhs ) const
    {
        return ( aUndirEdge == rhs.aUndirEdge ) && ( bUndirEdge == rhs.bUndirEdge );
    }
};

} //namespace MR
