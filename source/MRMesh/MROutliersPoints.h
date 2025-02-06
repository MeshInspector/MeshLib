#pragma once
#include "MRMeshFwd.h"
#include "MRMesh/MRExpected.h"
#include "MRUnionFind.h"
#include "MRBitSet.h"

namespace MR
{

struct OutlierParams
{
    int maxClusterSize = 0;
    int numNeigbors = 0;
    float minHeight = 0;
    float minAngle = 0;
};

struct OutlierType
{
    enum Bit
    {
        SmallComponents = 1 << 0,
        WeaklyConnected = 1 << 1,
        FarSurface = 1 << 2,
        AwayNormal = 1 << 3,
        All = ( 1 << 4 ) - 1
    };
};

using OutlierTypeMask = uint32_t;


class FindOutliers
{
public:
    FindOutliers() = default;

    std::string prepare( const PointCloud& pc, float radius, OutlierTypeMask mask, ProgressCallback progress = {} ); // calculate caches

    void setParams( const OutlierParams& params ); // just set params

    Expected<VertBitSet> find( OutlierTypeMask mask, ProgressCallback progress = {} ); // unite and calculate actual outliers

private:
    Expected<VertBitSet> findSmallComponents( ProgressCallback progress = {} );
    Expected<VertBitSet> findWeaklyConnected( ProgressCallback progress = {} );
    Expected<VertBitSet> findFarSurface( ProgressCallback progress = {} );
    Expected<VertBitSet> findAwayNormal( ProgressCallback progress = {} );

    float radius_;
    OutlierParams params_;
    OutlierParams paramsOld_;

    UnionFind<VertId> unionFindStructure_;
    std::vector<uint8_t> weaklyConnectedStat_;
    std::vector<float> farSurfaceStat_;
    std::vector<float> badNormalStat_;

    OutlierTypeMask maskCached_ = 0; // true means valid cache

    VertBitSet validPoints_;
};

struct FindOutliersParams
{
    OutlierParams finderParams;
    float radius;

    OutlierTypeMask mask;

    ProgressCallback cb;
};

Expected<VertBitSet> findOutliers( const PointCloud& pc, const FindOutliersParams& params )
{
    FindOutliers finder;
    finder.prepare( pc, params.radius, params.mask );
    return finder.find( params.mask );
}

}
