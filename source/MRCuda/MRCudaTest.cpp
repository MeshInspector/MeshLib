#include "MRCudaTest.h"
#include "MRCudaBasic.h"
#include "MRCudaTest.cuh"
#include "MRMesh/MRImage.h"
#include "cuda_runtime.h"

namespace MR
{

namespace Cuda
{
void negatePicture( Image& image )
{
    cudaSetDevice( 0 );

    DynamicArray<uint8_t> cudaArray;
    cudaArray.fromBytes( ( const uint8_t* )image.pixels.data(), image.pixels.size() * sizeof( MR::Color ) );

    negatePictureKernel( cudaArray );
    cudaArray.toBytes( ( uint8_t* )image.pixels.data() );
}
}

}