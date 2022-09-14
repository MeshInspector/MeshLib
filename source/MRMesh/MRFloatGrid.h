#pragma once
#ifndef MRMESH_NO_VOXEL
// this header includes the whole OpenVDB, so please include it from .cpp files only

#include "MRMeshFwd.h"
#include "MRPch/MROpenvdb.h"

namespace MR
{

/**
 * \defgroup BasicStructuresGroup Basic Structures
 * \brief This chapter represents documentation about basic structures elements
 * \{
 */

/// this class just hides very complex type of typedef openvdb::FloatGrid
struct OpenVdbFloatGrid : openvdb::FloatGrid
{
    OpenVdbFloatGrid() noexcept = default;
    OpenVdbFloatGrid( openvdb::FloatGrid && in ) : openvdb::FloatGrid( std::move( in ) ) {}
};

inline openvdb::FloatGrid & ovdb( OpenVdbFloatGrid & v ) { return v; }
inline const openvdb::FloatGrid & ovdb( const OpenVdbFloatGrid & v ) { return v; }

/// makes MR::FloatGrid shared pointer taking the contents of the input pointer
MRMESH_API FloatGrid MakeFloatGrid( openvdb::FloatGrid::Ptr&& );

/// resample this grid to fit voxelScale
MRMESH_API FloatGrid resampled( const FloatGrid & grid, float voxelScale );

/// resample this grid to fit voxelScale
MRMESH_API FloatGrid resampled( const FloatGrid & grid, const Vector3f& voxelScale );

/// sets given region voxels value
/// \note region is in grid space (0 voxel id is minimum active voxel in grid)
MRMESH_API void setValue( FloatGrid & grid, const VoxelBitSet& region, float value );

/// \}

}
#endif
