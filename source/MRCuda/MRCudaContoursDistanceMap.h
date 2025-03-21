#pragma once

#include "exports.h"

#include "MRMesh/MRDistanceMap.h"

namespace MR::Cuda
{

/// Computes distance of 2d contours according to ContourToDistanceMapParams (works correctly only when withSign==false)
MRCUDA_API Expected<DistanceMap> distanceMapFromContours( const Polyline2& polyline, const ContourToDistanceMapParams& params );

/// Computes memory consumption of distanceMapFromContours function
MRCUDA_API size_t distanceMapFromContoursHeapBytes( const Polyline2& polyline, const ContourToDistanceMapParams& params );

} // namespace MR::Cuda
