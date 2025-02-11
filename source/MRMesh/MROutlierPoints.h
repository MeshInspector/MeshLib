#pragma once
#include "MRMeshFwd.h"
#include "MRMesh/MRExpected.h"
#include "MRUnionFind.h"
#include "MRBitSet.h"
#include "MRConstants.h"
#include "MRFlagOperators.h"

namespace MR
{

/// Parameters of various criteria for detecting outlier points
struct OutlierParams
{
    /// Maximum points in the outlier component
    int maxClusterSize = 20;
    /// Maximum number of adjacent points for an outlier point
    int maxNeighbors = 7;
    /// Minimum distance (as proportion of search radius) to the approximate surface from outliers point
    float minHeight = 0.3f;
    /// Minimum angle of difference of normal at outlier points
    /// @note available only if there are normals
    float minAngle = PI_F / 3.f;
};

/// Types of outlier points
enum class OutlierTypeMask
{
    SmallComponents = 1 << 0, ///< Small groups of points that are far from the rest
    WeaklyConnected = 1 << 1, ///< Points that have too few neighbors within the radius
    FarSurface = 1 << 2, ///< Points far from the surface approximating the nearest points
    AwayNormal = 1 << 3, ///< Points whose normals differ from the average norm of their nearest neighbors
    All = SmallComponents | WeaklyConnected | FarSurface | AwayNormal
};
MR_MAKE_FLAG_OPERATORS( OutlierTypeMask )

/// A class for searching for outliers of points
/// @detail The class caches the prepared search results, which allows to speed up the repeat search (while use same radius)
class MRMESH_CLASS OutliersDetector
{
public:
    OutliersDetector() = default;

    /// Make a preliminary stage of outlier search. Caches the result
    /// 
    /// @param pc point cloud
    /// @param radius radius of the search for neighboring points for analysis
    /// @param mask mask of the types of outliers that are looking for
    /// @param progress progress callback function
    /// @return error text or nothing
    MRMESH_API Expected<void> prepare( const PointCloud& pc, float radius, OutlierTypeMask mask, ProgressCallback progress = {} ); // calculate caches

    /// Set search parameters
    MRMESH_API void setParams( const OutlierParams& params );
    /// Get search parameters
    MRMESH_API const OutlierParams& getParams() const { return params_; }

    /// Make an outlier search based on preliminary data
    /// @param mask mask of the types of outliers you are looking for
    MRMESH_API Expected<VertBitSet> find( OutlierTypeMask mask, ProgressCallback progress = {} ); // unite and calculate actual outliers

private:
    Expected<VertBitSet> findSmallComponents( ProgressCallback progress = {} );
    Expected<VertBitSet> findWeaklyConnected( ProgressCallback progress = {} );
    Expected<VertBitSet> findFarSurface( ProgressCallback progress = {} );
    Expected<VertBitSet> findAwayNormal( ProgressCallback progress = {} );

    float radius_ = 1.f;
    OutlierParams params_;

    // Cached data
    UnionFind<VertId> unionFindStructure_; // SmallComponents
    std::vector<uint8_t> weaklyConnectedStat_; // WeaklyConnected
    std::vector<float> farSurfaceStat_; // FarSurface
    std::vector<float> badNormalStat_; // AwayNormal

    OutlierTypeMask maskCached_ = OutlierTypeMask( 0 ); // true means valid cache

    VertBitSet validPoints_;
};

/// Outlier point search parameters
struct FindOutliersParams
{
    OutlierParams finderParams; ///< Parameters of various criteria for detecting outlier points
    float radius = 1.f; ///< Radius of the search for neighboring points for analysis

    OutlierTypeMask mask = OutlierTypeMask::All; ///< Mask of the types of outliers that are looking for

    ProgressCallback progress = {}; ///< Progress callback
};

/// Finding outlier points
MRMESH_API Expected<VertBitSet> findOutliers( const PointCloud& pc, const FindOutliersParams& params );

}
