#pragma once

#include "MRVoxelsFwd.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRPartMapping.h"
#include "MRMesh/MRVector3.h"
#include <optional>

namespace MR
{

/**
 * \struct MR::MergeVolumePartSettings
 * \brief Parameters' structure for MR::mergeVolumePart
 * \ingroup VoxelGroup
 *
 * \sa \ref mergeVolumePart
 */
struct MergeVolumePartSettings
{
    /// callback to process the generated mesh before the side cutting, e.g. fixing specific generation artifacts
    using PreCutCallback = std::function<void( Mesh& mesh, float leftCutPosition, float rightCutPosition )>;
    PreCutCallback preCut = nullptr;
    /// callback to process the generated mesh after the side cutting, e.g. decimating
    using PostCutCallback = std::function<void( Mesh& )>;
    PostCutCallback postCut = nullptr;
    /// callback to process the destination mesh after merging, usually to map the generated mesh's faces/edges/vertices
    /// the second parameter is identical to the `mapping` field, except for one case:
    /// if the mapping is not initialized, only `src2tgtEdges` map will be provided (since it's used during processing)
    using PostMergeCallback = std::function<void( Mesh&, const PartMapping& )>;
    PostMergeCallback postMerge = nullptr;
    /// mapping with initialized maps required for the `postMerge` callback
    PartMapping mapping = {};
    /// origin (position of the (0;0;0) voxel) of the voxel volume part, usually specified for SimpleVolume
    Vector3f origin = {};
};

/**
 * \brief Merge one mesh with another generated from a voxel volume part
 * \details The helper function for generating a mesh from a voxel volume without full memory loading.
 * It performs several actions:
 *  - converts the voxel volume part into a mesh;
 *  - cuts it to make its sides accurate;
 *  - appends to the result mesh with the matching side cut contours.
 * The functions has requirements for the input parameters:
 *  - the result mesh must have side contours on `leftCutPosition`;
 *  - the voxel volume part must have enough data to generate a mesh with correct cut contours. The usual way is to make an overlap for each volume part.
 *
 * \sa \ref volumeToMeshByParts
 *
 * @param mesh - result mesh to which the generated mesh will be attached
 * @param cutContours - cut contours of the result mesh; must be placed at `leftCutPosition`
 * @param volume - voxel volume part
 * @param leftCutPosition - position on X axis where the left side of the generated mesh is cut; pass -FLT_MAX to omit a cut here
 * @param rightCutPosition - position on X axis where the right side of the generated mesh is cut; pass +FLT_MAX to omit a cut here
 * @param settings - additional parameters; see \ref MergeVolumePartSettings
 * @return nothing if succeeds, an error string otherwise
 */
template <typename Volume>
Expected<void>
mergeVolumePart( Mesh& mesh, std::vector<EdgePath>& cutContours, Volume&& volume, float leftCutPosition, float rightCutPosition,
                 const MergeVolumePartSettings& settings = {} );

/// functor returning a voxel volume part within the specified range, or an error string on failure
/// the offset parameter is also required for SimpleVolume parts
template <typename Volume>
using VolumePartBuilder = std::function<Expected<Volume> ( int begin, int end, std::optional<Vector3i>& offset )>;

/**
 * \struct MR::VolumeToMeshByPartsSettings
 * \brief Parameters' structure for MR::volumeToMeshByParts
 * \ingroup VoxelGroup
 *
 * \sa \ref volumeToMeshByParts
 */
struct VolumeToMeshByPartsSettings
{
    /// the upper limit of memory amount used to store a voxel volume part
    size_t maxVolumePartMemoryUsage = 2 << 28; // 256 MiB
    /// overlap in voxels between two parts
    size_t stripeOverlap = 4;
};

/**
 * \brief converts a voxel volume into a mesh without full memory loading
 *
 * \sa \ref mergeVolumePart
 *
 * @param builder - functor returning a voxel volume part within the specified range
 * @param dimensions - full voxel volume dimensions
 * @param voxelSize - voxel size used for mesh generation
 * @param settings - additional parameters; see \ref VolumeToMeshByPartsSettings
 * @param mergeSettings - additional parameters for merging function; see \ref MergeVolumePartSettings
 * @return a generated mesh or an error string
 */
template <typename Volume>
Expected<Mesh>
volumeToMeshByParts( const VolumePartBuilder<Volume>& builder, const Vector3i& dimensions, const Vector3f& voxelSize,
                     const VolumeToMeshByPartsSettings& settings = {}, const MergeVolumePartSettings& mergeSettings = {} );

}
