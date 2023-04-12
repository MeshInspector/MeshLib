#pragma once
#include "MRCudaBasic.cuh"

namespace MR
{

namespace Cuda
{

struct Color
{
    uint8_t r, g, b, a;
};

// call simple kernel that negate each pixel chanell in parallel
void negatePictureKernel( DynamicArray<Color>& data );

}

}