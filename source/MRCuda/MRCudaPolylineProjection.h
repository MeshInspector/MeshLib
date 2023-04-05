#pragma once
#include "exports.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRDistanceMap.h"

namespace MR { namespace Cuda
{

MRCUDA_API DistanceMap distanceMapFromContours( const MR::Polyline2& polyline, const ContourToDistanceMapParams& params );

}}