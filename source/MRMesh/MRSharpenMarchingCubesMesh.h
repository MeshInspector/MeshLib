#pragma once

#include "MRMeshFwd.h"

namespace MR
{

struct SharpenMarchingCubesMeshSettings
{
    /// minimal surface deviation to introduce new vertex in a voxel;
    /// recommended set equal to ( voxel size / 25 )
    float minNewVertDev = 0;
    /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes);
    /// recommended set equal to ( 5 * voxel size )
    float maxNewRank2VertDev = 0;
    /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes);
    /// recommended set equal to ( 2 * voxel size )
    float maxNewRank3VertDev = 0;
    /// relative to reference mesh
    float offset = 0;
    /// correct positions of the input vertices using reference mesh by not more than this distance;
    /// big correction can be wrong and result from self-intersections in the reference mesh
    /// recommended set equal to ( voxel size / 2 )
    float maxOldVertPosCorrection = 0;
    /// minimum dot product of two normals of neighbor triangles created during sharpening
    float minNormalDot = -0.9f;
    /// the number of iterations to best select positions for new vertices,
    /// the probability of self-intersections and spikes are higher if posSelIters = 0
    int posSelIters = 3;
    /// if non-null then created sharp edges will be saved here
    UndirectedEdgeBitSet * outSharpEdges = nullptr;
};

/// adjust the mesh \param vox produced by marching cubes method (NOT dual marching cubes) by
/// 1) correcting positions of all vertices to given offset relative to \param ref mesh (if correctOldVertPos == true);
/// 2) introducing new vertices in the voxels where the normals change abruptly.
/// \param face2voxel mapping from Face Id to Voxel Id where it is located
MRMESH_API void sharpenMarchingCubesMesh( const Mesh & ref, Mesh & vox, Vector<VoxelId, FaceId> & face2voxel,
    const SharpenMarchingCubesMeshSettings & settings );

} //namespace MR
