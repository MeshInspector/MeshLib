#pragma once

#include "MRMeshFwd.h"

namespace MR
{

struct SharpenMarchingCubesMeshSettings
{
    // minimal surface deviation to introduce new vertex in a voxel;
    // recommended set equal to ( voxel size / 25 )
    float newVertDev = 0;
    // relative to reference mesh
    float offset = 0;
    // correct positions of the input vertices using reference mesh
    bool correctOldVertPos = true;
    // minimum dot product of two normals of neighbor triangles created during sharpening
    float minNormalDot = -0.9f;
};

/// adjust the mesh \param vox produced by marching cubes method (NOT dual marching cubes) by
/// 1) correcting positions of all vertices to given offset relative to \param ref mesh (if correctOldVertPos == true);
/// 2) introducing new vertices in the voxels where the normals change abruptly.
/// \param face2voxel mapping from Face Id to Voxel Id where it is located
MRMESH_API void sharpenMarchingCubesMesh( const Mesh & ref, Mesh & vox, Vector<VoxelId, FaceId> & face2voxel,
    const SharpenMarchingCubesMeshSettings & settings );

} //namespace MR
