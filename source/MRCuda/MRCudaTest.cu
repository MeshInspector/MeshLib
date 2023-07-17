#include "MRCudaTest.cuh"
#include "device_launch_parameters.h"


namespace MR
{

namespace Cuda
{

__global__ void negateKernel( uint8_t* imagePtr, const int size )
{
    int pixelShift = blockIdx.x * blockDim.x + threadIdx.x;
    if ( pixelShift >= size )
        return;

    int ind = 4 * pixelShift;

    imagePtr[ind] = 255 - imagePtr[ind];
    imagePtr[ind + 1] = 255 - imagePtr[ind + 1];
    imagePtr[ind + 2] = 255 - imagePtr[ind + 2];
}

void negatePictureKernel( DynamicArray<Color>& data )
{
    constexpr int maxThreadsPerBlock = 640;
    int numBlocks = ( int( data.size() ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
    // kernel
    negateKernel<<< numBlocks, maxThreadsPerBlock >>>( ( uint8_t* )data.data(), int( data.size() ) );
}

}
}