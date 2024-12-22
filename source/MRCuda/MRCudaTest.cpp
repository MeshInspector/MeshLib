#include "MRCudaTest.h"
#include "MRCudaBasic.h"
#include "MRCudaTest.cuh"
#include "MRMesh/MRImage.h"
#include "cuda_runtime.h"

namespace MR
{

namespace Cuda
{
Expected<void> negatePicture( Image& image )
{
    if ( auto code = CUDA_EXEC( cudaSetDevice( 0 ) ) )
        return unexpected( Cuda::getError( code ) );

    DynamicArray<Cuda::Color> cudaArray;
    if ( auto code = cudaArray.fromVector( image.pixels ) )
        return unexpected( Cuda::getError( code ) );

    negatePictureKernel( cudaArray );
    if ( auto code = CUDA_EXEC( cudaGetLastError() ) )
        return unexpected( Cuda::getError( code ) );

    if ( auto code = cudaArray.toVector( image.pixels ) )
        return unexpected( Cuda::getError( code ) );
    return {};
}

} //namespace Cuda

} //namespace MR
