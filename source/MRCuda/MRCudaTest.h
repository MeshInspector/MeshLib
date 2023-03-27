#pragma once
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"

// This file exists for testing some cuda features

namespace MR
{

namespace Cuda
{

// This function inverts Color value (255 - value in each channel except alpha) 
MRCUDA_API void negatePicture( MR::Image& image );

}

}