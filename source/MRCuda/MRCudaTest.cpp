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
    CUDA_EXEC( cudaSetDevice( 0 ) );

    DynamicArray<Cuda::Color> cudaArray;
    cudaArray.fromVector( image.pixels );

    negatePictureKernel( cudaArray );
    CUDA_EXEC( cudaGetLastError() );

    cudaArray.toVector( image.pixels );
}

} //namespace Cuda

} //namespace MR
