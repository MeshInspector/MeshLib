#pragma once

#include "MRMeshFwd.h"
#include "MRMeshPart.h"
#include "MRProgressCallback.h"

namespace MR
{
struct SimpleVolume;
// closed surface is required
// surfaceOffset - number voxels around surface to calculate distance in (should be positive)
MRMESH_API FloatGrid meshToLevelSet( const MeshPart& mp, const AffineXf3f& xf,
                                     const Vector3f& voxelSize, float surfaceOffset = 3,
                                     const ProgressCallback& cb = {} );

// does not require closed surface, resulting grid cannot be used for boolean operations,
// surfaceOffset - the number voxels around surface to calculate distance in (should be positive)
MRMESH_API FloatGrid meshToDistanceField( const MeshPart& mp, const AffineXf3f& xf,
                                          const Vector3f& voxelSize, float surfaceOffset = 3,
                                          const ProgressCallback& cb = {} );

// make FloatGrid from SimpleVolume
// make copy of data
// grid can be used to make iso-surface later with gridToMesh function
MRMESH_API FloatGrid simpleVolumeToDenseGrid( const SimpleVolume& simpleVolue,
                                              const ProgressCallback& cb = {} );

// isoValue - layer of grid with this value would be converted in mesh
// isoValue can be negative only in level set grids
// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones 
//                       (curvature can be lost on high values)
MRMESH_API Mesh gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
                            float isoValue = 0.0f, float adaptivity = 0.0f,
                            const ProgressCallback& cb = {} );

}
