#include "MRCudaTest.h"
#include "MRCudaTest.cuh"

#include "MRCuda.cuh"
#include "MRCudaBasic.h"

#include "MRMesh/MRImage.h"

namespace MR
{

namespace Cuda
{
Expected<void> negatePicture( Image& image )
{
    CUDA_LOGE_RETURN_UNEXPECTED( cudaSetDevice( 0 ) );

    DynamicArray<Cuda::Color> cudaArray;
    CUDA_LOGE_RETURN_UNEXPECTED( cudaArray.fromVector( image.pixels ) );

    negatePictureKernel( cudaArray );
    CUDA_LOGE_RETURN_UNEXPECTED( cudaGetLastError() );

    CUDA_LOGE_RETURN_UNEXPECTED( cudaArray.toVector( image.pixels ) );

    return {};
}

void loadMRCudaDll()
{
}

} //namespace Cuda

} //namespace MR
