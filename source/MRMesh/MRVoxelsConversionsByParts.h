#pragma once
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRPartMapping.h"

namespace MR
{

/**
 * \struct MR::MergeGridPartSettings
 * \brief Parameters' structure for MR::mergeGridPart
 * \ingroup VoxelGroup
 *
 * \sa \ref mergeGridPart
 */
struct MergeGridPartSettings
{
    /// callback to process the generated mesh before the side cutting, e.g. fixing specific generation artifacts
    using PreCutCallback = std::function<void( Mesh& mesh, float leftCutPosition, float rightCutPosition )>;
    PreCutCallback preCut = nullptr;
    /// callback to process the generated mesh after the side cutting, e.g. decimating
    using PostCutCallback = std::function<void( Mesh& )>;
    PostCutCallback postCut = nullptr;
    /// callback to process the destination mesh after merging, usually to map the generated mesh's faces/edges/vertices
    using PostMergeCallback = std::function<void( Mesh&, const PartMapping& )>;
    PostMergeCallback postMerge = nullptr;
    /// mapping with initialized maps required for the `postMerge` callback
    /// if the mapping is not initialized, only `src2tgtEdges` map will be provided (since it's used during processing)
    PartMapping mapping = {};
};

/**
 * \brief Merge one mesh with another generated from a voxel grid part
 * \details The helper function for generating a mesh from a voxel grid without full memory loading.
 * It performs several actions:
 *  - converts the voxel grid part into a mesh;
 *  - cuts it to make its sides accurate;
 *  - appends to the result mesh with the matching side cut contours.
 * The functions has requirements for the input parameters:
 *  - the result mesh must have side contours on `leftCutPosition`;
 *  - the voxel grid part must have enough data to generate a mesh with correct cut contours. The usual way is to make an overlap for each grid part.
 *
 * \sa \ref gridToMeshByParts
 *
 * @param mesh - result mesh to which the generated mesh will be attached
 * @param cutContours - cut contours of the result mesh; must be placed at `leftCutPosition`
 * @param grid - voxel grid part
 * @param voxelSize - voxel size
 * @param leftCutPosition - position on X axis where the left side of the generated mesh is cut; pass -FLT_MAX to omit a cut here
 * @param rightCutPosition - position on X axis where the right side of the generated mesh is cut; pass +FLT_MAX to omit a cut here
 * @param settings - additional parameters; see \ref MergeGridPartSettings
 * @return nothing if succeeds, an error string otherwise
 */
MRMESH_API
VoidOrErrStr
mergeGridPart( Mesh& mesh, std::vector<EdgePath>& cutContours, FloatGrid&& grid, const Vector3f& voxelSize,
               float leftCutPosition, float rightCutPosition, const MergeGridPartSettings& settings = {} );

/// functor returning a voxel grid part within the specified range, or an error string on failure
using GridPartBuilder = std::function<tl::expected<FloatGrid, std::string> ( size_t begin, size_t end )>;

/**
 * \struct MR::GridToMeshByPartsSettings
 * \brief Parameters' structure for MR::gridToMeshByParts
 * \ingroup VoxelGroup
 *
 * \sa \ref gridToMeshByParts
 */
struct GridToMeshByPartsSettings
{
    /// the upper limit of memory amount used to store a voxel grid part
    size_t maxGridPartMemoryUsage = 2 << 28; // 256 MiB
    /// overlap in voxels between two parts
    size_t stripeOverlap = 3;
};

/**
 * \brief converts a voxel grid into a mesh without full memory loading
 *
 * \sa \ref mergeGridPart
 *
 * @param builder - functor returning a voxel grid part within the specified range
 * @param dimensions - full voxel grid dimensions
 * @param voxelSize - voxel size used for mesh generation
 * @param settings - additional parameters; see \ref GridToMeshByPartsSettings
 * @param mergeSettings - additional parameters for merging function; see \ref MergeGridPartSettings
 * @return a generated mesh or an error string
 */
MRMESH_API
tl::expected<Mesh, std::string>
gridToMeshByParts( const GridPartBuilder& builder, const Vector3i& dimensions, const Vector3f& voxelSize,
                   const GridToMeshByPartsSettings& settings = {}, const MergeGridPartSettings& mergeSettings = {} );

}
#endif
