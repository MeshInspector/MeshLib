#pragma once

#include "MRMeshFwd.h"
#include "MRBitSet.h"

namespace MR
{

struct MarkedContour3f
{
    Contour3f contour;
    BitSet marks; ///< subset of indices of contour field
};

/// \return marked contour with all points from (in) marked
[[nodiscard]] inline MarkedContour3f markedContour( Contour3f in )
{
    const auto sz = in.size();
    return MarkedContour3f{ .contour = std::move( in ), .marks = BitSet( sz, true ) };
}

/// \param in input marked contour
/// \param maxStep maximum distance from not-marked point of returned contour to a neighbor point
/// \return contour with same marked points and not-marked points located on input contour with given maximum distance along it
[[nodiscard]] MRMESH_API MarkedContour3f resampled( const MarkedContour3f & in, float maxStep );

} //namespace MR
