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

} //namespace MR
