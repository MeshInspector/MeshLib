#pragma once

#include "MRExtractIsolines.h"

namespace MR
{

/// returns a set of globally smooth lines (linear within each triangle),
/// passing via given set of points
MRMESH_API IsoLines getSmoothLinesViaVertices( const Mesh & mesh, const VertBitSet & vs );

} //namespace MR
