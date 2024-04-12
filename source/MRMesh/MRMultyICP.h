#pragma once
#include "MRICP.h"


namespace MR
{

class MRMESH_CLASS MeshOrPointsTag;
using MeshOrPointsId = Id<MeshOrPointsTag>;

struct MultyICPObject
{
    MeshOrPoints meshOrPoints;
    AffineXf3f xf;
};

using IndexedPairs = Vector<PointPairs, MeshOrPointsId>;

class MRMESH_CLASS MultyICP
{
public:
    MRMESH_API MultyICP( const Vector<MultyICPObject, MeshOrPointsId>& objects, float samplingVoxelSize );
    
    [[nodiscard]] MRMESH_API Vector<AffineXf3f, MeshOrPointsId> calculateTransformations();
    
    /// select pairs with origin samples on all objects
    MRMESH_API void resamplePoints( float samplingVoxelSize );

    /// tune algorithm params before run calculateTransformations()
    void setParams( const ICPProperties& prop ) { prop_ = prop; }
    [[nodiscard]] const ICPProperties& getParams() const { return prop_; }

    /// computes root-mean-square deviation between points
    [[nodiscard]] MRMESH_API float getMeanSqDistToPoint() const;

    /// computes root-mean-square deviation from points to target planes
    [[nodiscard]] MRMESH_API float getMeanSqDistToPlane() const;

    /// computes the number of active point pairs
    [[nodiscard]] MRMESH_API size_t getNumActivePairs() const;
private:
    
    Vector<MultyICPObject, MeshOrPointsId> objs_;
    Vector<IndexedPairs, MeshOrPointsId> pairsPerObj_;
    ICPProperties prop_;

    ICPExitType resultType_{ ICPExitType::NotStarted };

    /// in each pair updates the target data and performs basic filtering (activation)
    void updatePointPairs_();

    /// deactivate pairs that does not meet farDistFactor criterion
    void deactivatefarDistPairs_();

    int iter_ = 0;
    bool p2ptIter_();
    bool p2plIter_();
};

}