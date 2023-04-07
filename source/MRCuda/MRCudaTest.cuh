#pragma once
#include "MRCudaBasic.h"

namespace MR
{

namespace Cuda
{

// call simple kernel that negate each pixel chanell in parallel
void negatePictureKernel( DynamicArray<uint8_t>& data );

}

}