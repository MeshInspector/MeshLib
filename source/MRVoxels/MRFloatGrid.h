#pragma once
#include "MRVoxelsFwd.h"

// this is a lightweight header unlike MRVDBFloatGrid.h

#include "MRMesh/MRProgressCallback.h"

namespace MR
{

/**
 * \defgroup BasicStructuresGroup Basic Structures
 * \brief This chapter represents documentation about basic structures elements
 * \{
 */

struct MRVOXELS_CLASS OpenVdbFloatGrid;
class Histogram;

/// wrapper class that helps mrbind to avoid excess MRVDBFloatGrid.h includes
class MRVOXELS_CLASS FloatGrid
{
public:
    MRVOXELS_API FloatGrid();
    MRVOXELS_API FloatGrid( std::shared_ptr<OpenVdbFloatGrid> ptr );

    MRVOXELS_API void reset() noexcept;
    MRVOXELS_API void swap( FloatGrid& other ) noexcept;
    MRVOXELS_API static FloatGrid deepCopy( const FloatGrid& other ) noexcept;

    MRVOXELS_API OpenVdbFloatGrid* get() const noexcept;
    MRVOXELS_API OpenVdbFloatGrid& operator *() const noexcept;
    MRVOXELS_API OpenVdbFloatGrid* operator ->() const noexcept;

    MRVOXELS_API explicit operator bool() const noexcept;

    MRVOXELS_API std::shared_ptr<OpenVdbFloatGrid> toVdb() const noexcept;

private:
    std::shared_ptr<OpenVdbFloatGrid> ptr_;
};

/// returns the amount of heap memory occupied by grid
[[nodiscard]] MRVOXELS_API size_t heapBytes( const FloatGrid& grid );

/// resample this grid to fit voxelScale
MRVOXELS_API FloatGrid resampled( const FloatGrid& grid, float voxelScale, ProgressCallback cb = {} );

/// resample this grid to fit voxelScale
MRVOXELS_API FloatGrid resampled( const FloatGrid& grid, const Vector3f& voxelScale, ProgressCallback cb = {} );

/// returns cropped grid
MRVOXELS_API FloatGrid cropped( const FloatGrid& grid, const Box3i& box, ProgressCallback cb = {} );

/// returns number of velxes in the grid with pred(value) == true
[[nodiscard]] MRVOXELS_API size_t countVoxelsWithValuePred( const FloatGrid& grid, const std::function<bool( float )>& pred );

/// returns number of voxels in the grid with value less than given
[[nodiscard]] MRVOXELS_API size_t countVoxelsWithValueLess( const FloatGrid& grid, float value );

/// returns number of voxels in the grid with value greater than given
[[nodiscard]] MRVOXELS_API size_t countVoxelsWithValueGreater( const FloatGrid& grid, float value );

/// computes histogram of grid values with given range and number of bins
[[nodiscard]] MRVOXELS_API Histogram calculateHistogram( const FloatGrid& grid, float min, float max, size_t binsNumber, ProgressCallback cb = {} );

/// returns grid with gaussian filter applied
MRVOXELS_API void gaussianFilter( FloatGrid& grid, int width, int iters, ProgressCallback cb = {} );
MRVOXELS_API FloatGrid gaussianFiltered( const FloatGrid& grid, int width, int iters, ProgressCallback cb = {} );

/// returns the value at given voxel
[[nodiscard]] MRVOXELS_API float getValue( const FloatGrid & grid, const Vector3i & p );

/// sets given region voxels value
/// \note region is in grid space (0 voxel id is minimum active voxel in grid)
MRVOXELS_API void setValue( FloatGrid& grid, const Vector3i& p, float value );

/// sets given region voxels value
/// \note region is in grid space (0 voxel id is minimum active voxel in grid)
MRVOXELS_API void setValue( FloatGrid & grid, const VoxelBitSet& region, float value );

/// returns bounding box of active voxels in grid
/// min: including
/// max: excluding
[[nodiscard]] MRVOXELS_API Box3i findActiveBounds( const FloatGrid& grid );

/// returns dimensions of the bounding box of active voxels in grid
[[nodiscard]] MRVOXELS_API Vector3i findActiveDims( const FloatGrid& grid );

/// activates voxels of the grid within given box and deactivates all other voxels
/// \note box is in grid space, max: excluding
MRVOXELS_API void setActiveBounds( FloatGrid& grid, const Box3i& box, ProgressCallback cb = {} );

/// sets given region voxels value one by one
/// \note region is in grid space (0 voxel id is minimum active voxel in grid)
MRVOXELS_API void setValues( FloatGrid& grid, const VoxelBitSet& region, const std::vector<float>& values );

/// sets type of this grid as LEVEL SET (for normal flipping)
MRVOXELS_API void setLevelSetType( FloatGrid & grid );

/// classification of grid values, mirrors openvdb::GridClass
enum class FloatGridClass
{
    Unknown = 0,
    LevelSet,
    FogVolume,
    Staggered
};

/// returns the class of the grid
[[nodiscard]] MRVOXELS_API FloatGridClass getGridClass( const FloatGrid& grid );

/// sets the class of the grid
MRVOXELS_API void setGridClass( FloatGrid& grid, FloatGridClass gridClass );

/// returns the background value of the grid (the value of voxels not stored in it)
[[nodiscard]] MRVOXELS_API float background( const FloatGrid& grid );

/// returns the number of active voxels in the grid
[[nodiscard]] MRVOXELS_API size_t activeVoxelCount( const FloatGrid& grid );

/// union operation on two signed distance fields
/// \note this operation consumes FloatGrid b
MRVOXELS_API FloatGrid operator += ( FloatGrid & a, FloatGrid&& b );

/// difference operation on two signed distance fields
/// \note this operation consumes FloatGrid b
MRVOXELS_API FloatGrid operator -= ( FloatGrid & a, FloatGrid&& b );

/// intersection operation on two signed distance fields
/// \note this operation consumes FloatGrid b
MRVOXELS_API FloatGrid operator *= ( FloatGrid & a, FloatGrid&& b );

/// union operation on two signed distance fields
/// \note this operation returns new FloatGrid keeping a,b untouched
MRVOXELS_API FloatGrid operator + ( const FloatGrid& a, const FloatGrid& b );

/// difference operation on two signed distance fields
/// \note this operation returns new FloatGrid keeping a,b untouched
MRVOXELS_API FloatGrid operator - ( const FloatGrid& a, const FloatGrid& b );

/// intersection operation on two signed distance fields
/// \note this operation returns new FloatGrid keeping a,b untouched
MRVOXELS_API FloatGrid operator * ( const FloatGrid& a, const FloatGrid& b );

/// \}

}
