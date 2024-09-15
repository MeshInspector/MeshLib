#pragma once

#include "MRVoxelsFwd.h"
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRBox.h>

namespace MR
{

/// shift of zero voxel in 3D space and dimensions of voxel-grid
struct OriginAndDimensions
{
    Vector3f origin;
    Vector3i dimensions;
};

/// computes origin and dimensions of voxel-grid to cover given 3D box with given spacing (voxelSize)
inline OriginAndDimensions calcOriginAndDimensions( const Box3f & box, float voxelSize )
{
    const auto expansion = Vector3f::diagonal( 2 * voxelSize );
    const auto origin = box.min - expansion;
    return
    {
        .origin = origin,
        .dimensions = Vector3i( ( box.max + expansion - origin ) / voxelSize ) + Vector3i::diagonal( 1 )
    };
}

} //namespace MR

