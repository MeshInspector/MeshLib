//#pragma once
//#include "MRMeshFwd.h"
//#include "MRMesh/MRExpected.h"
//#include "MRUnionFind.h"
//#include "MRBitSet.h"
//
//namespace MR
//{
//
//struct OutlierParams
//{
//    int maxClusterSize = 0;
//    int numNeigbors = 0;
//    float height = 0;
//    float angle = 0;
//};
//
//enum OutlierTypeBit
//{
//    Away = 1,
//    Far = 2,
//    Small = 4
//};
//
//using OutlierTypeMask = uint32_t;
//
//
//class FindOutliers
//{
//public:
//    FindOutliers() = default;
//
//    std::string prepare( const PointCloud& pc, float radius, OutlierTypeMask mask ); // calculate caches
//
//    void setParams( const OutlierParams& params ); // just set params
//
//    Expected<VertBitSet> find( OutlierTypeMask mask ); // unite and calculate actual outliers
//
//private:
//    float radius_;
//    UnionFind<VertId> unionFindStructure_;
//    std::vector<uint8_t> weaklyConnectedStat_;
//    std::vector<float> farSurfaceStat_;
//    std::vector<float> badNormalStat_;
//
//    OutlierTypeMask mask_; // true means valid cache
//    OutlierParams params_;
//
//    VertBitSet region_;
//};
//
//struct FindOutliersParams
//{
//    OutlierParams finderParams;
//    float radius;
//
//    OutlierTypeMask mask;
//
//    ProgressCallback cb;
//};
//
//Expected<VertBitSet> findOutliers( const PointCloud& pc, const FindOutliersParams& params )
//{
//    FindOutliers finder;
//    finder.prepare( pc, params.radius, params.mask );
//    return finder.find( params.mask );
//}
//
//}
