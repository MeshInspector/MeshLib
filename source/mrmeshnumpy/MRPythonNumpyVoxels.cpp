#ifndef MESHLIB_NO_VOXELS
#include "MRPython/MRPython.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRParallelMinMax.h"
#include "MRVoxels/MRVoxelsVolume.h"

MR::SimpleVolumeMinMax simpleVolumeFrom3Darray( const pybind11::buffer& voxelsArray )
{
    pybind11::buffer_info info = voxelsArray.request();
    if ( info.ndim != 3 )
        throw std::runtime_error( "shape of input python vector 'voxelsArray' should be (x,y,z)" );

    MR::SimpleVolumeMinMax res;
    res.dims = MR::Vector3i( int( info.shape[0] ), int( info.shape[1] ), int( info.shape[2] ) );
    size_t countPoints = size_t( res.dims.x ) * res.dims.y * res.dims.z;
    res.data.resize( countPoints );

    auto strideX = info.strides[0] / info.itemsize;
    auto strideY = info.strides[1] / info.itemsize;
    auto strideZ = info.strides[2] / info.itemsize;

    const size_t cX = res.dims.x;
    const size_t cXY = res.dims.x * res.dims.y;

    if ( info.format == pybind11::format_descriptor<double>::format() )
    {
        double* data = reinterpret_cast< double* >( info.ptr );
        for ( size_t x = 0; x < res.dims.x; ++x )
            for ( size_t y = 0; y < res.dims.y; ++y )
                for ( size_t z = 0; z < res.dims.z; ++z )
                    res.data[MR::VoxelId( x + y * cX + z * cXY )] = float( data[x * strideX + y * strideY + z * strideZ] );
    }
    else if ( info.format == pybind11::format_descriptor<float>::format() )
    {
        float* data = reinterpret_cast< float* >( info.ptr );
        for ( size_t x = 0; x < res.dims.x; ++x )
            for ( size_t y = 0; y < res.dims.y; ++y )
                for ( size_t z = 0; z < res.dims.z; ++z )
                    res.data[MR::VoxelId( x + y * cX + z * cXY )] = data[x * strideX + y * strideY + z * strideZ];
    }
    else
        throw std::runtime_error( "dtype of input python vector should be float32 or float64" );

    std::tie( res.min, res.max ) = MR::parallelMinMax( res.data );
    return res;
}

pybind11::array_t<double> getNumpy3Darray( const MR::SimpleVolume& simpleVolume )
{
    using namespace MR;
    // Allocate and initialize some data;
    const size_t size = size_t( simpleVolume.dims.x ) * simpleVolume.dims.y * simpleVolume.dims.z;
    double* data = new double[size];

    const size_t cX = simpleVolume.dims.x;
    const size_t cXY = simpleVolume.dims.x * simpleVolume.dims.y;
    const size_t cZ = simpleVolume.dims.z;
    const size_t cZY = simpleVolume.dims.z * simpleVolume.dims.y;

    for ( size_t x = 0; x < simpleVolume.dims.x; ++x )
        for ( size_t y = 0; y < simpleVolume.dims.y; ++y )
            for ( size_t z = 0; z < simpleVolume.dims.z; ++z )
                data[x * cZY + y * cZ + z] = simpleVolume.data[VoxelId( x + y * cX + z * cXY )];

    // Create a Python object that will free the allocated
    // memory when destroyed:
    pybind11::capsule freeWhenDone( data, [] ( void* f )
    {
        bool* data = reinterpret_cast< bool* >( f );
        delete[] data;
    } );

    return pybind11::array_t<double>(
        { simpleVolume.dims.x, simpleVolume.dims.y, simpleVolume.dims.z }, // shape
        { simpleVolume.dims.y * simpleVolume.dims.z * sizeof( double ), simpleVolume.dims.z * sizeof( double ), sizeof( double ) }, // C-style contiguous strides for bool
        data, // the data pointer
        freeWhenDone ); // numpy array references this parent
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshnumpy, VoxelsVolumeNumpyConvert, [] ( pybind11::module_& m )
{
    m.def( "simpleVolumeFrom3Darray", &simpleVolumeFrom3Darray, pybind11::arg( "3DvoxelsArray" ),
        "Convert numpy 3D array to SimpleVolume" );
    m.def( "getNumpy3Darray", &getNumpy3Darray, pybind11::arg( "simpleVolume" ),
        "Convert SimpleVolume to numpy 3D array" );
} )
#endif
