#pragma once

#include "MRPrecisePredicates3.h"

namespace MR
{

/// \ingroup MathGroup
/// \{

/// given line segment and two triangles known to intersect it, finds the order of intersection using precise predicates:
/// true: segm[0], segm ^ ta, segm ^ tb, segm[1]
/// false: segm[0], segm ^ tb, segm ^ ta, segm[1]
[[nodiscard]] MRMESH_API bool segmentIntersectionOrder(
    const PreciseVertCoords segm[2],
    const PreciseVertCoords ta[3],
    const PreciseVertCoords tb[3] );

/// \}

} //namespace MR
