#include "MRVDBConversions.h"

#include "MRVDBFloatGrid.h"
#include "MROpenVDB.h"
#include "MRVoxelsVolume.h"
#include "MRVoxelsVolumeAccess.h"
#include "MRMesh/MRTimer.h"

#include <cfloat>

namespace MR
{

constexpr float denseVolumeToGridTolerance = 1e-6f;

VdbVolume floatGridToVdbVolume( FloatGrid grid )
{
    if ( !grid )
        return {};
    MR_TIMER;
    VdbVolume res;
    evalGridMinMax( grid, res.min, res.max );
    auto dim = grid->evalActiveVoxelDim();
    res.dims = Vector3i( dim.x(), dim.y(), dim.z() );
    res.data = std::move( grid );
    return res;
}

template <>
void putSimpleVolumeInDenseGrid(
        openvdb::FloatGrid& grid,
        const Vector3i& minCoord, const SimpleVolume& simpleVolume, ProgressCallback cb
    )
{
    MR_TIMER;
    if ( cb )
        cb( 0.0f );
    openvdb::math::Coord dimsCoord( simpleVolume.dims.x, simpleVolume.dims.y, simpleVolume.dims.z );
    openvdb::math::CoordBBox denseBBox( toVdb( minCoord ), toVdb( minCoord ) + dimsCoord.offsetBy( -1 ) );
    openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense( denseBBox, const_cast< float* >( simpleVolume.data.data() ) );
    if ( cb )
        cb( 0.5f );
    openvdb::tools::copyFromDense( dense, grid, denseVolumeToGridTolerance );
    if ( cb )
        cb( 1.f );
}

template <>
void putSimpleVolumeInDenseGrid(
        FloatGrid& grid,
        const Vector3i& minCoord, const SimpleVolume& simpleVolume, ProgressCallback cb
    )
{
    openvdb::FloatGrid& gridRef = *grid;
    putSimpleVolumeInDenseGrid( gridRef, minCoord, simpleVolume, cb );
}

template <typename VolumeType>
void putVolumeInDenseGrid(
        openvdb::FloatGrid::Accessor& gridAccessor,
        const Vector3i& minCoord, const VolumeType& volume, ProgressCallback cb )
{
    MR_TIMER;
    if ( cb )
        cb( 0.0f );

    VoxelsVolumeAccessor<VolumeType> volumeAccessor( volume );

    for ( int z = 0; z < volume.dims.z; ++z )
    {
        if ( !reportProgress( cb, ( float )z / ( float )volume.dims.z ) )
            return;
        for ( int y = 0; y < volume.dims.y; ++y )
        {
            for ( int x = 0; x < volume.dims.x; ++x )
            {
                auto loc = Vector3i{ x, y, z };
                auto coord = toVdb( minCoord + loc );
                gridAccessor.setValue( coord, volumeAccessor.get( loc ) );
            }
        }
    }
}

template <>
void putSimpleVolumeInDenseGrid(
        openvdb::FloatGrid::Accessor& gridAccessor,
        const Vector3i& minCoord, const SimpleVolume& simpleVolume, ProgressCallback cb
    )
{
    putVolumeInDenseGrid( gridAccessor, minCoord, simpleVolume, cb );
}

FloatGrid simpleVolumeToDenseGrid( const SimpleVolume& simpleVolume,
                                   float background,
                                   ProgressCallback cb )
{
    MR_TIMER;
    std::shared_ptr<openvdb::FloatGrid> grid = std::make_shared<openvdb::FloatGrid>( FLT_MAX );
    putSimpleVolumeInDenseGrid( *grid, { 0, 0, 0 }, simpleVolume, cb );
    openvdb::tools::changeBackground( grid->tree(), background );
    return MakeFloatGrid( std::move( grid ) );
}

VdbVolume simpleVolumeToVdbVolume( const SimpleVolumeMinMax& simpleVolume, ProgressCallback cb /*= {} */ )
{
    VdbVolume res;
    res.data = simpleVolumeToDenseGrid( simpleVolume, simpleVolume.min, cb );
    res.dims = simpleVolume.dims;
    res.voxelSize = simpleVolume.voxelSize;
    res.min = simpleVolume.min;
    res.max = simpleVolume.max;
    return res;
}

VdbVolume functionVolumeToVdbVolume( const FunctionVolume& functoinVolume, ProgressCallback cb /*= {} */ )
{
    MR_TIMER;
    VdbVolume res;
    std::shared_ptr<openvdb::FloatGrid> grid = std::make_shared<openvdb::FloatGrid>( FLT_MAX );
    auto gridAccessor = grid->getAccessor();
    putVolumeInDenseGrid( gridAccessor, { 0, 0, 0 }, functoinVolume, cb );
    auto minMax = openvdb::tools::minMax( grid->tree() );
    res.min = minMax.min();
    res.max = minMax.max();
    openvdb::tools::changeBackground( grid->tree(), res.min );
    res.data = MakeFloatGrid( std::move( grid ) );
    res.dims = functoinVolume.dims;
    res.voxelSize = functoinVolume.voxelSize;

    return res;
}

} //namespace MR
