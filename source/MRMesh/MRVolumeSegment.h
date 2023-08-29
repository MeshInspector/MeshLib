#pragma once
#include "MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRVector3.h"
#include "MRExpected.h"
#include "MRSimpleVolume.h"
#include "MRBitSet.h"
#include "MRProgressCallback.h"
#include <string>

namespace MR
{

/**
 * \defgroup VolumeSegmentationGroup Volume (Voxel) Segmentation
 * \brief This chapter represents documentation about Volume (Voxel) Segmentation
 * \ingroup VoxelGroup
 * \{
 */

/**
 * \brief Creates mesh from voxels mask
 * \param mask in space of whole volume
 *  density inside mask is expected to be higher then outside
 */
MRMESH_API Expected<MR::Mesh, std::string> meshFromVoxelsMask( const VdbVolume& volume, const VoxelBitSet& mask );

 /**
  * \brief Parameters for volume segmentation
  * 
  * \sa \ref segmentVolume
  */
struct VolumeSegmentationParameters
{
    /// Exponent modifier of path building metric (paths are built between voxel pairs and then marked as tooth seed)
    float buildPathExponentModifier{-1.0f};
    /// Exponent modifier of graph cutting metric (volume presents graph with seeds, this graph are min cut)
    float segmentationExponentModifier{3000.0f};
    /// Segment box expansion (only part of volume are segmented, this parameter shows how much to expand this part)
    int voxelsExpansion{25};
};

/**
 * \brief Simple segment volume
 * \details
 * 1. Build paths between points pairs \n
 * 2. Mark paths as inside part seeds \n
 * 3. Mark volume part edges as outside part seeds \n
 * 4. Return mesh from segmented inside part
 */
MRMESH_API Expected<MR::Mesh, std::string> segmentVolume( const VdbVolume& volume, const std::vector<std::pair<Vector3f, Vector3f>>& pairs,
                                                              const VolumeSegmentationParameters& params = VolumeSegmentationParameters() );

struct VoxelMetricParameters;

/**
 * \brief Class for voxels segmentation
 *
 * <table border=0> <caption id="VolumeSegmenter_examples"></caption>
 * <tr> <td> \image html voxel_segmentation/voxel_segmentation_0_0.png "Before (a)" width = 350cm </td>
 *      <td> \image html voxel_segmentation/voxel_segmentation_0_1.png "Before (b)" width = 350cm </td> </tr>
 *      <td> \image html voxel_segmentation/voxel_segmentation_0_2.png "After" width = 350cm </td> </tr>
 * </table>
 */
class VolumeSegmenter
{
public:
    MRMESH_API VolumeSegmenter( const VdbVolume& volume );

    enum SeedType
    {
        Inside,
        Outside,
        Count
    };

    /// Builds path with given parameters, marks result as seedType seeds
    MRMESH_API void addPathSeeds( const VoxelMetricParameters& metricParameters, SeedType seedType, float exponentModifier = -1.0f );
    
    /// Reset seeds with given ones
    MRMESH_API void setSeeds( const std::vector<Vector3i>& seeds, SeedType seedType );
    
    /// Adds new seeds to stored
    MRMESH_API void addSeeds( const std::vector<Vector3i>& seeds, SeedType seedType );

    /// Return currently stored seeds
    MRMESH_API const std::vector<Vector3i>& getSeeds( SeedType seedType ) const;

    /// Segments volume, return inside part segmentation (VoxelBitSet in space of VolumePart)
    MRMESH_API Expected<VoxelBitSet, std::string> segmentVolume( float segmentationExponentModifier = 3000.0f, int voxelsExpansion = 25, ProgressCallback cb = {} );
    
    /// Returns mesh of given segment
    MRMESH_API Expected<MR::Mesh, std::string> createMeshFromSegmentation( const VoxelBitSet& segmentation ) const;

    /// Dimensions of volume part, filled after segmentation
    MRMESH_API const Vector3i& getVolumePartDimensions() const;

    /// Min voxel of volume part box in whole volume space, filled after segmentation
    MRMESH_API const Vector3i& getMinVoxel() const;
private:
    const VdbVolume& volume_;

    SimpleVolume volumePart_;

    Vector3i minVoxel_;
    Vector3i maxVoxel_;

    std::array<std::vector<Vector3i>, size_t( SeedType::Count )> seeds_;
    std::array<VoxelBitSet, size_t( SeedType::Count )> seedsInVolumePartSpace_;

    bool seedsChanged_{true};

    void setupVolumePart_( int voxelsExpansion );
};

/// \}

}
#endif
