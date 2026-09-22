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
    /// explicitly define ctors to avoid warning C5267: definition of implicit copy constructor is deprecated because it has a user-provided destructor
    IComputePointsToDistanceVolume() = default;
    IComputePointsToDistanceVolume( const IComputePointsToDistanceVolume& ) = default;
    IComputePointsToDistanceVolume( IComputePointsToDistanceVolume&& ) noexcept = default;

    IComputePointsToDistanceVolume & operator = ( const IComputePointsToDistanceVolume& ) = default;
    IComputePointsToDistanceVolume & operator = ( IComputePointsToDistanceVolume&& ) noexcept = default;

    virtual ~IComputePointsToDistanceVolume() = default;

    /// whether this implementation can process given input, e.g. it fits in GPU memory
    virtual bool canCompute( const PointCloud& cloud, const PointsToDistanceVolumeParams& params ) const = 0;

    /// makes the whole volume at once
    virtual Expected<SimpleVolumeMinMax> compute( const PointCloud& cloud, const PointsToDistanceVolumeParams& params ) const = 0;
};

/// complements \ref IComputePointsToDistanceVolume with computation in z-slabs, which needs less memory
class MRVOXELS_CLASS IComputePointsToDistanceVolumeByParts : public IComputePointsToDistanceVolume
{
public:
    /// gets one z-slab of the volume, starting at z-layer zOffset
    using AddPartFunc = std::function<Expected<void>( const SimpleVolumeMinMax& volume, int zOffset )>;

    /// makes the volume by z-slabs, passing each of them in addPart with layerOverlap layers shared by neighbours
    virtual Expected<void> computeByParts( const PointCloud& cloud, const PointsToDistanceVolumeParams& params,
        AddPartFunc addPart, int layerOverlap ) const = 0;
};

/// complements \ref IComputePointsToDistanceVolume with a lazily evaluated volume,
/// which needs the least memory since the consumer evaluates only the voxels it reads
class MRVOXELS_CLASS IComputePointsToDistanceFunctionVolume : public IComputePointsToDistanceVolume
{
public:
    /// makes a volume evaluated on demand; it is valid as long as cloud and params are alive
    virtual FunctionVolume computeFunctionVolume( const PointCloud& cloud, const PointsToDistanceVolumeParams& params ) const = 0;
};

/// CPU implementation of IComputePointsToDistanceVolume, used by pointsToMeshFusion when no other one is given;
/// prefer its lazy computeFunctionVolume over compute, which materializes the whole volume in memory
class MRVOXELS_CLASS ComputePointsToDistanceVolume : public IComputePointsToDistanceFunctionVolume
{
public:
    // see methods' descriptions in the interfaces above
    MRVOXELS_API bool canCompute( const PointCloud& cloud, const PointsToDistanceVolumeParams& params ) const override;
    MRVOXELS_API Expected<SimpleVolumeMinMax> compute( const PointCloud& cloud, const PointsToDistanceVolumeParams& params ) const override;
    MRVOXELS_API FunctionVolume computeFunctionVolume( const PointCloud& cloud, const PointsToDistanceVolumeParams& params ) const override;
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
