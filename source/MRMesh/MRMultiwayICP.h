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
    
    [[nodiscard]] MRMESH_API Vector<AffineXf3f, MeshOrPointsId> calculateTransformations( ProgressCallback cb = {} );
    
    /// select pairs with origin samples on all objects
    MRMESH_API void resamplePoints( float samplingVoxelSize );

    /// in each pair updates the target data and performs basic filtering (activation)
    MRMESH_API void updateAllPointPairs();

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

    /// if in independent equations mode - creates separate equation system for each object
    /// otherwise creates single large equation system for all objects
    bool devIndependentEquationsModeEnabled() const { return maxGroupSize_ == 1; }
    void devEnableIndependentEquationsMode( bool on ) { maxGroupSize_ = on ? 1 : 0; }

    /// returns status info string
    [[nodiscard]] MRMESH_API std::string getStatusInfo() const; 
private:
    Vector<MeshOrPointsXf, MeshOrPointsId> objs_;
    Vector<IndexedPairs, MeshOrPointsId> pairsPerObj_;
    ICPProperties prop_;

    ICPExitType resultType_{ ICPExitType::NotStarted };

    std::function<void( int )> perIterationCb_;

    /// reserves memory for all pairs
    /// if currently in cascade mode (objs.size() > maxGroupSize_) reserves only for pairs inside groups
    void reservePairs_( const Vector<VertBitSet, MeshOrPointsId>& samples );
    /// updates pairs among same groups only
    void updatePointsPairsGroupWise_();
    /// deactivate pairs that does not meet farDistFactor criterion
    void deactivatefarDistPairs_();

    /// returns unified samples for all objects in group
    std::vector<Vector3f> resampleGroup_( MeshOrPointsId groupFirst, MeshOrPointsId groupLastEx );

    using Level = int;
    class GroupTag;
    using GroupId = Id<GroupTag>;
    struct GroupPair
    {

    };

    float samplingSize_{ 0.0f };
    // this parameter indicates maximum number of objects that might be aligned simultaneously in multi-way mode
    // N<0 - means all of the objects
    // N=1 - means that all objects are aligned independently
    // N>1 - means that registration is applied cascade with N grouping step:
    //       1) separate all objects to groups with N members
    //       2) align objects inside these groups
    //       3) join groups and create groups of these joined objects (2nd layer groups)
    //       4) align this groups
    // N>number of objects - same as 0
    int maxGroupSize_{ 64 };
    int iter_ = 0;
    bool doIteration_( bool p2pl );
    bool p2ptIter_();
    bool p2plIter_();
    bool multiwayIter_( bool p2pl = true );
    bool multiwayIter_( int groupSize, bool p2pl = true );
};

}