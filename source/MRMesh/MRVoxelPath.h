#pragma once
#ifndef __EMSCRIPTEN__
#include "MRMeshFwd.h"
#include <functional>

namespace MR
{
using VoxelsMetric = std::function<float( size_t from, size_t to )>;

enum QuarterBit : char
{
    LeftLeft = 0b1,
    LeftRight = 0b10,
    RightLeft = 0b100,
    RightRight = 0b1000,
    All = 0b1111
};

// Plane of slice in which to find path
enum SlicePlain
{
    YZ, // = 0 cause main axis is x - [0]
    ZX, // = 1 cause main axis is y - [1]
    XY, // = 2 cause main axis is z - [2]
    None // special value not to limit path in one slice 
};

// Parameters for building metric function
struct VoxelMetricParameters
{
    size_t start; // start voxel index
    size_t stop;  // stop voxel index 
    float maxDistRatio{1.5f}; // max distance ratio: if (dist^2(next,start) + dist^2(next,stop) > maxDistRatio^2*dist^2(start,stop)) - candidate is not processed 
    SlicePlain plane{None}; // if not None - builds path in one slice of voxels (make sure start and stop has same main axis coordinate)
    char quatersMask{QuarterBit::All}; // quarter of building path, if plane is selected, it should be (LeftLeft | LeftRigth) or (RigthLeft | RightRight) or All
};

// e^(modifier*(dens1+dens2))
[[nodiscard]] MRMESH_API VoxelsMetric voxelsExponentMetric( const ObjectVoxels& voxels, const VoxelMetricParameters& parameters,
                                                           float modifier = -1.0f );

// sum of dense differences with start and stop voxels
[[nodiscard]] MRMESH_API VoxelsMetric voxelsSumDiffsMetric( const ObjectVoxels& voxels, const VoxelMetricParameters& parameters );

// builds shortest path in given metric from start to finish voxels; if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API std::vector<size_t> buildSmallestMetricPath( const ObjectVoxels & voxels, 
                                                                     const VoxelsMetric & metric,
                                                                     size_t start, size_t finish );

}
#endif
