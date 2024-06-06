#pragma once
#include "MRICP.h"
#include "MRGridSampling.h"

namespace MR
{

using ICPObjects = Vector<MeshOrPointsXf, ObjId>;

using ICPLayer = int;
class ICPElemtTag;
using ICPElementId = Id<ICPElemtTag>;
struct ICPGroupPair : public ICPPairData
{
    ObjVertId srcId;
    ObjVertId tgtClosestId;
};
struct ICPGroupPairs : public IPointPairs
{
    virtual const ICPPairData& operator[]( size_t idx ) const override { return vec[idx]; }
    virtual ICPPairData& operator[]( size_t idx ) override { return vec[idx]; }
    std::vector<ICPGroupPair> vec;
};

using ICPGroupProjector = std::function<void( const Vector3f& p, MeshOrPoints::ProjectionResult& res, ObjId& resId )>;
/// in each pair updates the target data and performs basic filtering (activation)
void updateGroupPairs( ICPGroupPairs& pairs, const ICPObjects& objs,
    ICPGroupProjector srcProjector, ICPGroupProjector tgtProjector,
    float cosTreshold, float distThresholdSq, bool mutualClosest );

using ICPPairsGrid = Vector<Vector<ICPGroupPairs, ICPElementId>, ICPElementId>;

/// This class allows you to register many objects having similar parts
/// and known initial approximations of orientations/locations using
/// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
class MRMESH_CLASS MultiwayICP
{
public:
    MRMESH_API MultiwayICP( const ICPObjects& objects, float samplingVoxelSize );
    
    /// runs ICP algorithm given input objects, transformations, and parameters;
    /// \return adjusted transformations of all objects to reach registered state
    [[nodiscard]] MRMESH_API Vector<AffineXf3f, ObjId> calculateTransformations( ProgressCallback cb = {} );
    
    /// select pairs with origin samples on all objects
    MRMESH_API void resamplePoints( float samplingVoxelSize );

    /// in each pair updates the target data and performs basic filtering (activation)
    /// in cascade mode only useful for stats update 
    MRMESH_API void updateAllPointPairs();

    /// tune algorithm params before run calculateTransformations()
    void setParams( const ICPProperties& prop ) { prop_ = prop; }
    [[nodiscard]] const ICPProperties& getParams() const { return prop_; }

    /// computes root-mean-square deviation between points
    [[nodiscard]] MRMESH_API float getMeanSqDistToPoint() const;

    /// computes root-mean-square deviation from points to target planes
    [[nodiscard]] MRMESH_API float getMeanSqDistToPlane() const;

    /// computes the number of active point pairs
    [[nodiscard]] MRMESH_API size_t getNumActivePairs() const;

    /// sets callback that will be called for each iteration
    void setPerIterationCallback( std::function<void( int inter )> callback ) { perIterationCb_ = std::move( callback ); }

    /// if in independent equations mode - creates separate equation system for each object
    /// otherwise creates single large equation system for all objects
    bool devIndependentEquationsModeEnabled() const { return maxGroupSize_ == 1; }
    void devEnableIndependentEquationsMode( bool on ) { maxGroupSize_ = on ? 1 : 0; }

    /// returns status info string
    [[nodiscard]] MRMESH_API std::string getStatusInfo() const;

private:
    ICPObjects objs_;
    Vector<ICPPairsGrid, ICPLayer> pairsGridPerLayer_;
    ICPProperties prop_;

    ICPExitType resultType_{ ICPExitType::NotStarted };

    std::function<void( int )> perIterationCb_;

    /// reserves space in pairsGridPerLayer_ according to mode and GroupIndexer
    void setupLayers_();

    /// reserves memory for all pairs
    /// if currently in cascade mode (objs.size() > maxGroupSize_) reserves only for pairs inside groups
    void reservePairsLayer0_( Vector<VertBitSet, ObjId>&& samples );

    using LayerSamples = Vector<Vector<MultiObjsSamples, ICPElementId>, ICPLayer>;
    LayerSamples resampleUpperLayers_();
    void reserveUpperLayerPairs_( LayerSamples&& samples );

    /// calculates and updates pairs 2nd and next steps of cascade mode
    void updateLayerPairs_( ICPLayer l );
    /// deactivate pairs that does not meet farDistFactor criterion, for given layer
    void deactivateFarDistPairs_( ICPLayer l );

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
    bool cascadeIter_( bool p2pl = true );
};

}