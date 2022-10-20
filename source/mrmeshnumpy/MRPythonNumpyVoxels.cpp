#include "MRMesh/MRPython.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRSimpleVolume.h"
#include "MRMesh/MRVector3.h"

MR::SimpleVolume simpleVolumeFrom3Darray( const pybind11::buffer& voxelsArray )
{
    pybind11::buffer_info info = voxelsArray.request();
    if ( info.ndim != 3 )
    {
        PyErr_SetString( PyExc_RuntimeError, "shape of input python vector 'voxelsArray' should be (x,y,z)" );
        assert( false );
    }

    MR::SimpleVolume res;
    res.dims = MR::Vector3i( int( info.shape[0] ), int( info.shape[1] ), int( info.shape[2] ) );
    size_t countPoints = res.dims.x * res.dims.y * res.dims.z;
    res.data.resize( countPoints );

    if ( info.format == pybind11::format_descriptor<double>::format() )
    {
        double* data = reinterpret_cast< double* >( info.ptr );
        for ( size_t i = 0; i < countPoints; ++i )
            res.data[i] = float( data[i] );
    }
    else if ( info.format == pybind11::format_descriptor<float>::format() )
    {
        float* data = reinterpret_cast< float* >( info.ptr );
        for ( size_t i = 0; i < countPoints; ++i )
            res.data[i] = data[i];
    }
    else
    {
        PyErr_SetString( PyExc_RuntimeError, "dtype of input python vector should be float32 or float64" );
        assert( false );
    }

    return res;
}

pybind11::array_t<double> getNumpy3Darray( const MR::SimpleVolume& simpleVolume )
{
    using namespace MR;
    // Allocate and initialize some data;
    const size_t size = simpleVolume.dims.x * simpleVolume.dims.y * simpleVolume.dims.z;
    double* data = new double[size];
    for ( size_t i = 0; i < size; ++i )
        data[i] = simpleVolume.data[i];

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
