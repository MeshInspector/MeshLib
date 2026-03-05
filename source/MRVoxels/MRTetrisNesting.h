#pragma once
#include "MRVoxelsFwd.h"
#include "MRMesh/MRNestingStructures.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRVolumeIndexer.h"


namespace MR
{

namespace Nesting
{

struct TetrisDensifyOptions
{
    /// size of block for tetris box
    float voxelSize{ 0.0f };

    /// tetris box will be densify in these directions one by one
    std::vector<OutEdge> densificationSequence = { OutEdge::MinusZ,OutEdge::MinusY,OutEdge::MinusX };

    ProgressCallback cb;

    Vector<ObjId, VoxelId>* nestVoxelsCache{ nullptr }; ///< [in/out] pre-allocated voxels vector (to speedup allocation)
    Vector3i* nestDimensionsCache{ nullptr }; ///< [in/out] dimensions of the nest (complimentary to voxels data)
    VoxelBitSet* occupiedVoxelsCache{ nullptr }; ///< [in/out] voxels that blocks movement of floating (input) meshes (to provide input and output occupancy status)
};

struct TetrisDensifyParams
{
    NestingBaseParams baseParams;
    TetrisDensifyOptions options;
};

/// make nested meshes more compact by representing them via voxels and pushing to nest zero
MRVOXELS_API Expected<Vector<AffineXf3f, ObjId>> tetrisNestingDensify( const Vector<MeshXf, ObjId>& meshes, const TetrisDensifyParams& params );

}

}