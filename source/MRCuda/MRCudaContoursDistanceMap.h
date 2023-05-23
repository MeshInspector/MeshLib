#pragma once
#include "exports.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRDistanceMap.h"

namespace MR { namespace Cuda
{

/// Computes distance of 2d contours according to ContourToDistanceMapParams (works correctly only when withSign==false)
MRCUDA_API DistanceMap distanceMapFromContours( const MR::Polyline2& polyline, const ContourToDistanceMapParams& params );

/// Computes memory consumption of distanceMapFromContours function
MRCUDA_API size_t distanceMapFromContoursHeapBytes( const MR::Polyline2& polyline, const ContourToDistanceMapParams& params );
}}