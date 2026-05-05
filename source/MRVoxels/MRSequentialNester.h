#pragma once
#include "MRMesh/MRBoxNesting.h"
#include "MRTetrisNesting.h"
#include "MRMesh/MRBitSet.h"

namespace MR
{

namespace Nesting
{

/// class to add meshes to nest sequentially 
class MRVOXELS_CLASS SequentialNester
{
public:
    /// if voxelSize > 0 peform densification on each addition
    MRVOXELS_API SequentialNester( const NestingBaseParams& params, float voxelSize );

    /// tries to add single mesh to the nest
    /// returns true if mesh is nested, false otherwise (can be canceled)
    MRVOXELS_API Expected<NestingResult> nestMesh( const MeshXf& meshXf, const BoxNestingOptions& options, const std::vector<OutEdge>* densificationSequence = nullptr );

    /// tries to add multiple mesh to the nest
    /// returns bitset of nested meshes (can be canceled)
    MRVOXELS_API Expected<Vector<NestingResult, ObjId>> nestMeshes( const Vector<MeshXf, ObjId>& meshes, const BoxNestingOptions& options, const std::vector<OutEdge>* densificationSequence = nullptr );
private:
    NestingBaseParams baseParams_;
    TetrisDensifyOptions tetrisOptions_;

    Vector<ObjId, VoxelId> voxelsCache_;
    Vector3i dimensionsCache_;
    VoxelBitSet occupiedVoxelsCache_;

    std::vector<Box3f> nestedBoxesCache_;
    std::vector<BoxNestingCorner> addBoxCornersCache_;
};

}

}