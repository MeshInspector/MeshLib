#ifndef MESHLIB_NO_VOXELS
#include "MRPython/MRPython.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRParallelMinMax.h"
#include "MRVoxels/MRVoxelsVolume.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRBitSetParallelFor.h"

using namespace MR;

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

    VolumeIndexer indexer( res.dims );
    if ( info.format == pybind11::format_descriptor<double>::format() )
    {
        double* data = reinterpret_cast< double* >( info.ptr );
        ParallelFor( 0_vox, indexer.endId(), [&] ( VoxelId i )
        {
            auto pos = indexer.toPos( i );
            res.data[i] = float( data[size_t( pos.x ) * strideX + size_t( pos.y ) * strideY + size_t( pos.z ) * strideZ] );
        } );
    }
    else if ( info.format == pybind11::format_descriptor<float>::format() )
    {
        float* data = reinterpret_cast< float* >( info.ptr );
        ParallelFor( 0_vox, indexer.endId(), [&] ( VoxelId i )
        {
            auto pos = indexer.toPos( i );
            res.data[i] = data[size_t( pos.x ) * strideX + size_t( pos.y ) * strideY + size_t( pos.z ) * strideZ];
        } );
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

    const size_t cZ = simpleVolume.dims.z;
    const size_t cZY = simpleVolume.dims.z * simpleVolume.dims.y;
    VolumeIndexer indexer( simpleVolume.dims );
    ParallelFor( 0_vox, indexer.endId(), [&] ( VoxelId i )
    {
        auto pos = indexer.toPos( i );
        data[size_t( pos.x ) * cZY + size_t( pos.y ) * cZ + size_t( pos.z )] = simpleVolume.data[i];
    } );

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

VoxelBitSet voxelBitSetFrom3Darray( const pybind11::buffer& boolsArray )
{
    pybind11::buffer_info info = boolsArray.request();
    if ( info.ndim != 3 )
        throw std::runtime_error( "shape of input python vector 'boolsArray' should be (x,y,z)" );
    if ( info.format != pybind11::format_descriptor<bool>::format() )
        throw std::runtime_error( "dtype of input python vector should be bool" );

    const VolumeIndexer indexer( Vector3i( int( info.shape[0] ), int( info.shape[1] ), int( info.shape[2] ) ) );
    const auto strideX = info.strides[0] / info.itemsize;
    const auto strideY = info.strides[1] / info.itemsize;
    const auto strideZ = info.strides[2] / info.itemsize;
    const bool* data = reinterpret_cast< const bool* >( info.ptr );

    VoxelBitSet res( indexer.size() );
    BitSetParallelForAll( res, [&] ( VoxelId i )
    {
        const auto pos = indexer.toPos( i );
        if ( data[size_t( pos.x ) * strideX + size_t( pos.y ) * strideY + size_t( pos.z ) * strideZ] )
            res.set( i );
    } );
    return res;
}

pybind11::array_t<bool> getNumpy3DarrayFromVoxelBitSet( const VoxelBitSet& bitSet, const Vector3i& dims )
{
    const VolumeIndexer indexer( dims );
    if ( bitSet.size() > indexer.size() )
        throw std::runtime_error( "bitSet is larger than the volume with given dimensions" );

    pybind11::array_t<bool> res( { dims.x, dims.y, dims.z } );
    bool* data = res.mutable_data();
    const size_t cZ = dims.z;
    const size_t cZY = size_t( dims.z ) * dims.y;
    ParallelFor( 0_vox, indexer.endId(), [&] ( VoxelId i )
    {
        const auto pos = indexer.toPos( i );
        data[size_t( pos.x ) * cZY + size_t( pos.y ) * cZ + size_t( pos.z )] = bitSet.test( i );
    } );
    return res;
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshnumpy, VoxelsVolumeNumpyConvert, [] ( pybind11::module_& m )
{
    m.def( "simpleVolumeFrom3Darray", &simpleVolumeFrom3Darray, pybind11::arg( "3DvoxelsArray" ),
        "Convert numpy 3D array to SimpleVolume" );
    m.def( "getNumpy3Darray", &getNumpy3Darray, pybind11::arg( "simpleVolume" ),
        "Convert SimpleVolume to numpy 3D array" );
    m.def( "voxelBitSetFrom3Darray", &voxelBitSetFrom3Darray, pybind11::arg( "boolsArray" ),
        "Convert numpy 3D array of bools with shape (x,y,z) to VoxelBitSet indexed as in VolumeIndexer" );
    m.def( "getNumpy3Darray", &getNumpy3DarrayFromVoxelBitSet, pybind11::arg( "voxelBitSet" ), pybind11::arg( "dims" ),
        "Convert VoxelBitSet of the volume with given dimensions to numpy 3D array of bools with shape (x,y,z)" );
} )
#endif
