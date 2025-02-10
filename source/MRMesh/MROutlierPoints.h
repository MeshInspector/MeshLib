#pragma once
#include "MRMeshFwd.h"
#include "MRMesh/MRExpected.h"
#include "MRUnionFind.h"
#include "MRBitSet.h"

namespace MR
{

/// Parameters of various criteria for detecting outlier points
struct OutlierParams
{
    /// Maximum points in the outlier component
    int maxClusterSize = 0;
    /// Maximum number of adjacent points for an outlier point
    int maxNeighbors = 0;
    /// Minimum distance (as proportion of search radius) to the approximate surface from outliers point
    float minHeight = 0;
    /// Minimum angle of difference of normal at outlier points
    /// @note available only if there are normals
    float minAngle = 0;
};

/// Types of outlier points
struct OutlierType
{
    enum Bit
    {
        SmallComponents = 1 << 0, ///< Small groups of points that are far from the rest
        WeaklyConnected = 1 << 1, ///< Points that have too few neighbors within the radius
        FarSurface = 1 << 2, ///< Points far from the surface approximating the nearest points
        AwayNormal = 1 << 3, ///< Points whose normals differ from the average norm of their nearest neighbors
        All = ( 1 << 4 ) - 1
    };
};

using OutlierTypeMask = uint32_t;

/// A class for searching for outliers of points
/// @detail The class caches the prepared search results, which allows to speed up the repeat search (while use same radius)
class OutliersDetector
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
    std::string prepare( const PointCloud& pc, float radius, OutlierTypeMask mask, ProgressCallback progress = {} ); // calculate caches

    /// Set search parameters
    void setParams( const OutlierParams& params );
    /// Get search parameters
    const OutlierParams& getParams() const { return params_; }

    /// Make an outlier search based on preliminary data
    /// @param mask mask of the types of outliers you are looking for
    Expected<VertBitSet> find( OutlierTypeMask mask, ProgressCallback progress = {} ); // unite and calculate actual outliers

private:
    Expected<VertBitSet> findSmallComponents( ProgressCallback progress = {} );
    Expected<VertBitSet> findWeaklyConnected( ProgressCallback progress = {} );
    Expected<VertBitSet> findFarSurface( ProgressCallback progress = {} );
    Expected<VertBitSet> findAwayNormal( ProgressCallback progress = {} );

    float radius_;
    OutlierParams params_;

    // Cached data
    UnionFind<VertId> unionFindStructure_; // SmallComponents
    std::vector<uint8_t> weaklyConnectedStat_; // WeaklyConnected
    std::vector<float> farSurfaceStat_; // FarSurface
    std::vector<float> badNormalStat_; // AwayNormal

    OutlierTypeMask maskCached_ = 0; // true means valid cache

    VertBitSet validPoints_;
};

/// Outlier point search parameters
struct FindOutliersParams
{
    OutlierParams finderParams; ///< Parameters of various criteria for detecting outlier points
    float radius; ///< Radius of the search for neighboring points for analysis

    OutlierTypeMask mask; ///< Mask of the types of outliers that are looking for

    ProgressCallback cb; ///< Progress callback
};

/// Finding outlier points
Expected<VertBitSet> findOutliers( const PointCloud& pc, const FindOutliersParams& params )
{
    OutliersDetector finder;
    finder.prepare( pc, params.radius, params.mask );
    return finder.find( params.mask );
}

}
