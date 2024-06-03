#pragma once
#include "MRICP.h"
#include "MRGridSampling.h"

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
    MRMESH_API void updateAllPointPairs();

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
    bool devIndependentEquationsModeEnabled() const { return maxGroupSize_ == 1; }
    void devEnableIndependentEquationsMode( bool on ) { maxGroupSize_ = on ? 1 : 0; }

    /// returns status info string
    [[nodiscard]] MRMESH_API std::string getStatusInfo() const;

private:
    Vector<MeshOrPointsXf, ObjId> objs_;
    Vector<IndexedPairs, ObjId> pairsPerObj_;
    ICPProperties prop_;

    ICPExitType resultType_{ ICPExitType::NotStarted };

    std::function<void( int )> perIterationCb_;

    /// reserves memory for all pairs
    /// if currently in cascade mode (objs.size() > maxGroupSize_) reserves only for pairs inside groups
    void reservePairs_( const Vector<VertBitSet, ObjId>& samples );
    /// updates pairs among same groups only
    void updatePointsPairsGroupWise_();
    /// deactivate pairs that does not meet farDistFactor criterion
    void deactivatefarDistPairs_();

    using Layer = int;
    class GroupTag;
    using GroupId = Id<GroupTag>;
    struct GroupPoint
    {
        ObjVertId id;
        Vector3f point;
        Vector3f norm;
    };
    struct GroupPair
    {
        GroupPoint src;
        GroupPoint tgt;
        float weight{ 1.0f };
    };
    struct GroupPairs
    {
        std::vector<GroupPair> vec;
        BitSet active;
    };
    using IndexedGroupPairs = Vector<GroupPairs, GroupId>;
    using LayerPairs = Vector<Vector<IndexedGroupPairs, GroupId>, Layer>;
    

    // prepares data for cascade mode
    LayerPairs pairsPerLayer_;
    void resampleLayers_();
    void reserveLayerPairs_( const Vector<Vector<MultiObjsSamples, GroupId>, Layer>& samples );
    // calculates and updates pairs 2nd and next steps of cascade mode
    void updateLayerPairs_( Layer l );
    bool projectGroupPair_( GroupPair& pair, ObjId srcFirst, ObjId srcLast, ObjId tgtFirst, ObjId tgtLast );

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