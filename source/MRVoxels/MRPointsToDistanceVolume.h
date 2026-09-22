#pragma once

#include "MRVoxelsFwd.h"

#include "MRDistanceVolumeParams.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRPointCloud.h"

#include <functional>

namespace MR
{

struct PointsToDistanceVolumeParams : DistanceVolumeParams
{
    /// it the distance of highest influence of a point;
    /// the maximal influence distance is 3*sigma; beyond that distance the influence is strictly zero
    float sigma = 1;

    /// minimum sum of influence weights from surrounding points for a voxel to get a value, meaning that there shall be at least this number of points in close proximity
    float minWeight = 1;

    /// coefficient used for weight calculation: e^(dist^2 * -invSigmaModifier * sigma^-2)
    /// values: (0;inf)
    float invSigmaModifier = 0.5f;

    /// changes the way point angle affects weight, by default it is linearly increasing with dot product
    /// if enabled - increasing as dot product^(0.5) (with respect to its sign)
    bool sqrtAngleWeight{ false };

    /// optional input: if this pointer is set then function will use these normals instead of ones present in cloud
    const VertNormals* ptNormals = nullptr;
};

/// makes SimpleVolume filled with signed distances to points with normals
[[nodiscard]] MRVOXELS_API Expected<SimpleVolume> pointsToDistanceVolume( const PointCloud & cloud, const PointsToDistanceVolumeParams& params );

/// makes FunctionVolume representing signed distances to points with normals
[[nodiscard]] MRVOXELS_API FunctionVolume pointsToDistanceFunctionVolume( const PointCloud & cloud, const PointsToDistanceVolumeParams& params );

/// abstract class for computing a volume of signed distances to points with normals
class MRVOXELS_CLASS IComputePointsToDistanceVolume
{
public:
    virtual ~IComputePointsToDistanceVolume() = default;

    /// gets one z-slab of the volume, starting at z-layer zOffset
    using AddPartFunc = std::function<Expected<void>( const SimpleVolumeMinMax& volume, int zOffset )>;

    /// whether this implementation can process given input, e.g. it fits in GPU memory
    virtual bool canCompute( const PointCloud& cloud, const PointsToDistanceVolumeParams& params ) const = 0;

    /// makes the whole volume at once
    virtual Expected<SimpleVolumeMinMax> compute( const PointCloud& cloud, const PointsToDistanceVolumeParams& params ) const = 0;

    /// whether \ref computeByParts is implemented; it is preferred over \ref compute since it needs less memory
    virtual bool supportsByParts() const = 0;

    /// makes the volume by z-slabs, passing each of them in addPart with layerOverlap layers shared by neighbours
    virtual Expected<void> computeByParts( const PointCloud& cloud, const PointsToDistanceVolumeParams& params,
        AddPartFunc addPart, int layerOverlap ) const = 0;
};

/// CPU implementation of IComputePointsToDistanceVolume
class MRVOXELS_CLASS ComputePointsToDistanceVolume : public IComputePointsToDistanceVolume
{
public:
    // see methods' descriptions in IComputePointsToDistanceVolume
    MRVOXELS_API bool canCompute( const PointCloud& cloud, const PointsToDistanceVolumeParams& params ) const override;
    MRVOXELS_API Expected<SimpleVolumeMinMax> compute( const PointCloud& cloud, const PointsToDistanceVolumeParams& params ) const override;
    MRVOXELS_API bool supportsByParts() const override;
    MRVOXELS_API Expected<void> computeByParts( const PointCloud& cloud, const PointsToDistanceVolumeParams& params,
        AddPartFunc addPart, int layerOverlap ) const override;
};

/// given
/// \param cloud      a point cloud
/// \param colors     colors of each point in the cloud
/// \param tgtPoints  some target points
/// \param tgtVerts   mask of valid target points
/// \param sigma      the distance of highest influence of a point
/// \param cb         progress callback
/// computes the colors in valid target points by averaging the colors from the point cloud
[[nodiscard]] MRVOXELS_API Expected<VertColors> calcAvgColors( const PointCloud & cloud, const VertColors & colors,
    const VertCoords & tgtPoints, const VertBitSet & tgtVerts, float sigma, const ProgressCallback & cb = {} );

} //namespace MR
