#include "MRCudaTest.h"
#include "MRMesh/MRImage.h"
#include "cuda_runtime.h"
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

void negatePicture( Image& image )
{
    cudaSetDevice( 0 );
    uint8_t* cudaPointer{ nullptr };
    auto size = image.resolution.x * image.resolution.y;
    cudaMalloc( ( void** )&cudaPointer, size * sizeof( uint8_t ) * 4 );
    cudaMemcpy( cudaPointer, image.pixels.data(), size * sizeof( uint8_t ) * 4, cudaMemcpyHostToDevice );
    int maxThreadsPerBlock = 0;
    cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
    int numBlocks = ( size + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
    // kernel
    negateKernel << <numBlocks, maxThreadsPerBlock >> > ( cudaPointer, size );

    cudaMemcpy( image.pixels.data(), cudaPointer, size * sizeof( uint8_t ) * 4, cudaMemcpyDeviceToHost );
    cudaFree( cudaPointer );
}
}
}