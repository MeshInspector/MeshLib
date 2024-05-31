#pragma once

#include "MRICP.h"

namespace MR
{

using IndexedPairs = Vector<PointPairs, ObjId>;

/// This class allows you to register many objects having similar parts
/// and known initial approximations of orientations/locations using
/// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
class MRMESH_CLASS MultiwayICP
{
public:
    MRMESH_API MultiwayICP( const Vector<MeshOrPointsXf, ObjId>& objects, float samplingVoxelSize );
    
    /// runs ICP algorithm given input objects, transformations, and parameters;
    /// \return adjusted transformations of all objects to reach registered state
    [[nodiscard]] MRMESH_API Vector<AffineXf3f, ObjId> calculateTransformations( ProgressCallback cb = {} );
    
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
    [[nodiscard]] MRMESH_API float getMeanSqDistToPoint( ObjId id ) const;

    /// computes root-mean-square deviation from points to target planes
    [[nodiscard]] MRMESH_API float getMeanSqDistToPlane() const;

    /// computes root-mean-square deviation from points to target planes  of given object
    [[nodiscard]] MRMESH_API float getMeanSqDistToPlane( ObjId id ) const;

    /// computes the number of active point pairs
    [[nodiscard]] MRMESH_API size_t getNumActivePairs() const;

    /// computes the number of active point pairs of given object
    [[nodiscard]] MRMESH_API size_t getNumActivePairs( ObjId id ) const;

    /// sets callback that will be called for each iteration
    void setPerIterationCallback( std::function<void( int inter )> callback ) { perIterationCb_ = std::move( callback ); }

    /// if in independent equations mode - creates separate equation system for each object
    /// otherwise creates single large equation system for all objects
    bool independentEquationsModeEnabled() const { return independentEquationsMode_; }
    void enableIndependentEquationsMode( bool on ) { independentEquationsMode_ = on; }

    /// returns status info string
    [[nodiscard]] MRMESH_API std::string getStatusInfo() const;

private:
    Vector<MeshOrPointsXf, ObjId> objs_;
    Vector<IndexedPairs, ObjId> pairsPerObj_;
    ICPProperties prop_;

    ICPExitType resultType_{ ICPExitType::NotStarted };

    std::function<void( int )> perIterationCb_;

    /// deactivate pairs that does not meet farDistFactor criterion
    void deactivatefarDistPairs_();

    float samplingSize_{ 0.0f };
    bool independentEquationsMode_{ false };
    int iter_ = 0;
    bool p2ptIter_();
    bool p2plIter_();
    bool multiwayIter_( bool p2pl = true );
};

}