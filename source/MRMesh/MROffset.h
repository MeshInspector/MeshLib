#pragma once
#ifndef __EMSCRIPTEN__
#include "MRMeshFwd.h"
#include "MRMeshPart.h"
#include "MRProgressCallback.h"

namespace MR
{

// This struct represents parameters for offsetting with voxels conversions
struct OffsetParameters
{
    // Size of voxel in grid conversions
    // if value is negative, it is calculated automatically (mesh bounding box are divided to 5e6 voxels)
    float voxelSize{-1.0f};
    // Decimation ratio of result mesh [0..1], this is applied on conversion from voxels to mesh
    // note: it does not work good, better use common decimation after offsetting
    float adaptivity{0.0f};
    // Progress callback 
    ProgressCallback callBack{};
};

// Offsets mesh by converting it to voxels and back
// for non closed meshes do offset in both directions (normal and -normal)
// so result mesh is always closed
[[nodiscard]] MRMESH_API Mesh offsetMesh( const MeshPart& mp, float offset, const OffsetParameters& params = {} );

// Offsets polyline by converting it to voxels and back
// do offset in both directions (normal and -normal)
// so result mesh is always closed
[[nodiscard]] MRMESH_API Mesh offsetPolyline( const Polyline3& polyline, float offset, const OffsetParameters& params = {} );

}
#endif
