#pragma once

#include "MRICP.h"

namespace MR
{

class MRMESH_CLASS MeshOrPointsTag;
using MeshOrPointsId = Id<MeshOrPointsTag>;
using IndexedPairs = Vector<PointPairs, MeshOrPointsId>;

class MRMESH_CLASS MultiwayICP
{
public:
    MRMESH_API MultiwayICP( const Vector<MeshOrPointsXf, MeshOrPointsId>& objects, float samplingVoxelSize );
    
    [[nodiscard]] MRMESH_API Vector<AffineXf3f, MeshOrPointsId> calculateTransformations();
    
    /// select pairs with origin samples on all objects
    MRMESH_API void resamplePoints( float samplingVoxelSize );

    /// in each pair updates the target data and performs basic filtering (activation)
    MRMESH_API void updatePointPairs();

    /// tune algorithm params before run calculateTransformations()
    void setParams( const ICPProperties& prop ) { prop_ = prop; }
    [[nodiscard]] const ICPProperties& getParams() const { return prop_; }

    /// computes root-mean-square deviation between points
    [[nodiscard]] MRMESH_API float getMeanSqDistToPoint() const;

    /// computes root-mean-square deviation between points of given object
    [[nodiscard]] MRMESH_API float getMeanSqDistToPoint( MeshOrPointsId id ) const;

    /// computes root-mean-square deviation from points to target planes
    [[nodiscard]] MRMESH_API float getMeanSqDistToPlane() const;

    /// computes root-mean-square deviation from points to target planes  of given object
    [[nodiscard]] MRMESH_API float getMeanSqDistToPlane( MeshOrPointsId id ) const;

    /// computes the number of active point pairs
    [[nodiscard]] MRMESH_API size_t getNumActivePairs() const;

    /// computes the number of active point pairs of given object
    [[nodiscard]] MRMESH_API size_t getNumActivePairs( MeshOrPointsId id ) const;

    /// sets callback that will be called for each iteration
    void setPerIterationCallback( std::function<void( int inter )> callback ) { perIterationCb_ = std::move( callback ); }

    /// returns status info string
    [[nodiscard]] MRMESH_API std::string getStatusInfo() const; 
private:
    Vector<MeshOrPointsXf, MeshOrPointsId> objs_;
    Vector<IndexedPairs, MeshOrPointsId> pairsPerObj_;
    ICPProperties prop_;

    ICPExitType resultType_{ ICPExitType::NotStarted };

    std::function<void( int )> perIterationCb_;

    /// deactivate pairs that does not meet farDistFactor criterion
    void deactivatefarDistPairs_();

    int iter_ = 0;
    bool p2ptIter_();
    bool p2plIter_();
    bool multiwayIter_();
};

}