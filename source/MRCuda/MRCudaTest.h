#pragma once
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRExpected.h"

// This file exists for testing some cuda features

namespace MR
{

namespace Cuda
{

// This function inverts Color value (255 - value in each channel except alpha) 
MRCUDA_API Expected<void> negatePicture( MR::Image& image );

// call this function to load MRCuda shared library
MRCUDA_API void loadMRCudaDll();

}

}