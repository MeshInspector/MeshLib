#pragma once

#include "MRPch/MRBindingMacros.h"
#include "MRICPEnums.h"
#include "MRMeshOrPoints.h"
#include "MRMatrix3.h"
#include "MRId.h"
#include "MRConstants.h"
#include "MRAffineXf.h"
#include "MRBitSet.h"
#include <cfloat>

namespace MR
{

struct ICPPairData
{
    /// coordinates of the source point after transforming in world space
    Vector3f srcPoint;

    /// normal in source point after transforming in world space
    Vector3f srcNorm;

    /// coordinates of the closest point on target after transforming in world space
    Vector3f tgtPoint;

    /// normal in the target point after transforming in world space
    Vector3f tgtNorm;

    /// squared distance between source and target points
    float distSq = 0.f;

    /// weight of the pair (to prioritize over other pairs)
    float weight = 1.f;

    friend bool operator == ( const ICPPairData&, const ICPPairData& ) = default;
};

/// Stores a pair of points: one samples on the source and the closest to it on the target
struct PointPair : public ICPPairData
{
    /// id of the source point
    VertId srcVertId;

    /// for point clouds it is the closest vertex on target,
    /// for meshes it is the closest vertex of the triangle with the closest point on target
    VertId tgtCloseVert;

    /// cosine between normals in source and target points
    float normalsAngleCos = 1.f;

    /// true if if the closest point on target is located on the boundary (only for meshes)
    bool tgtOnBd = false;

    friend bool operator == ( const PointPair&, const PointPair& ) = default;
};

/// Simple interface for pairs holder
struct IPointPairs
{
    // required to declare explicitly to avoid warnings
    IPointPairs() = default;
    IPointPairs( const IPointPairs& ) = default;
    IPointPairs& operator=( const IPointPairs& ) = default;
    IPointPairs( IPointPairs&& ) noexcept = default;
    IPointPairs& operator=( IPointPairs&& ) noexcept = default;
    virtual ~IPointPairs() = default;

    virtual const ICPPairData& operator[]( size_t ) const = 0;
    virtual ICPPairData& operator[]( size_t ) = 0;
    virtual size_t size() const = 0;
    BitSet active; ///< whether corresponding pair from vec must be considered during minimization
};

struct PointPairs : public IPointPairs
{
    virtual const ICPPairData& operator[]( size_t idx ) const override { return vec[idx]; }
    virtual ICPPairData& operator[]( size_t idx ) override { return vec[idx]; }
    virtual size_t size() const override { return vec.size(); }
    std::vector<PointPair> vec; ///< vector of all point pairs both active and not
};

/// returns the number of samples able to form pairs
[[nodiscard]] inline size_t getNumSamples( const IPointPairs& pairs ) { return pairs.active.size(); }

/// computes the number of active pairs
[[nodiscard]] MRMESH_API size_t getNumActivePairs( const IPointPairs& pairs );

struct NumSum
{
    int num = 0;
    double sum = 0;

    friend NumSum operator + ( const NumSum & a, const NumSum & b ) { return { a.num + b.num, a.sum + b.sum }; }

    [[nodiscard]] float rootMeanSqF() const { return ( num <= 0 ) ? FLT_MAX : (float)std::sqrt( sum / num ); }
};

/// computes the number of active pairs and the sum of squared distances between points
/// or the difference between the squared distances between points and inaccuracy
[[nodiscard]] MRMESH_API NumSum getSumSqDistToPoint( const IPointPairs& pairs, std::optional<double> inaccuracy = {} );

/// computes the number of active pairs and the sum of squared deviation from points to target planes
/// or the difference between the squared distances between points to target planes and inaccuracy
[[nodiscard]] MRMESH_API NumSum getSumSqDistToPlane( const IPointPairs& pairs, std::optional<double> inaccuracy = {});

/// computes root-mean-square deviation between points
[[nodiscard]] inline float getMeanSqDistToPoint( const IPointPairs& pairs ) { return getSumSqDistToPoint( pairs ).rootMeanSqF(); }

/// computes root-mean-square deviation from points to target planes
[[nodiscard]] inline float getMeanSqDistToPlane( const IPointPairs& pairs ) { return getSumSqDistToPlane( pairs ).rootMeanSqF(); }

/// returns status info string
[[nodiscard]] MRMESH_API std::string getICPStatusInfo( int iterations, ICPExitType exitType );

/// given prepared (p2pl) object, finds the best transformation from it of given type with given limitations on rotation angle and global scale
[[nodiscard]] MRMESH_API AffineXf3d getAligningXf( const PointToPlaneAligningTransform & p2pl,
    ICPMode mode, float angleLimit, float scaleLimit, const Vector3f & fixedRotationAxis );

/// parameters of ICP algorithm
struct ICPProperties
{
    /// The method how to update transformation from point pairs, see description of each option in ICPMethod
    ICPMethod method = ICPMethod::PointToPlane;

    /// Rotation angle during one iteration of ICPMethod::PointToPlane will be limited by this value.
    /// This is to reduce possible instability.
    float p2plAngleLimit = PI_F / 6.0f; // [radians]

    /// Scaling during one iteration of ICPMethod::PointToPlane will be limited by this value.
    /// This is to reduce possible instability.
    float p2plScaleLimit = 2;

    /// If source and target points in a pair both have normals, then dot-product of normals must be not smaller than (cosThreshold),
    /// otherwise such point pair will be deactivated (ignored) during transformation computation.
    /// This is to get rid of erroneous pairs on unrelated parts of objects, which are close only by chance.
    float cosThreshold = 0.7f; // in [-1,1]

    /// The squared distance between source and target points in a pair must be not greater than (distThresholdSq),
    /// otherwise such point pair will be deactivated (ignored) during transformation computation.
    /// This is to get rid of outliers in input objects.
    float distThresholdSq = 1.f; // [distance^2]

    /// First, root-mean-square distance between points in all active pairs is computed as D.
    /// Then, the distance between source and target points in a pair must be not greater than (farDistFactor * D),
    /// otherwise such point pair will be deactivated (ignored) during transformation computation.
    /// This is to progressively reduce the distance threshold as the algorithm converge to a solution.
    float farDistFactor = 3.f; // dimensionless

    /// Selects the group of transformations, where to find a solution (e.g. with scaling or without, with rotation or without, ...).
    /// See the description of each option in ICPMode.
    ICPMode icpMode = ICPMode::AnyRigidXf;

    /// Additional parameter for ICPMode::OrthogonalAxis and ICPMode::FixedAxis transformation groups.
    Vector3f fixedRotationAxis;

    /// The maximum number of iterations that the algorithm can perform.
    /// Increase this parameter if you need a higher precision or if initial approximation is not very precise.
    int iterLimit = 10;

    /// The algorithm will stop before making all (iterLimit) iterations, if there were
    /// consecutive (badIterStopCount) iterations, during which the average distance between points in active pairs did not diminish.
    int badIterStopCount = 3;

    /// The algorithm will stop before making all (iterLimit) iterations,
    /// if the average distance between points in active pairs became smaller than this value.
    float exitVal = 0; // [distance]

    /// A pair of points is activated only if both points in the pair are mutually closest (reciprocity test passed),
    /// some papers recommend this mode for filtering out wrong pairs, but it can be too aggressive and deactivate (almost) all pairs.
    bool mutualClosest = false;
};

/// reset active bit if pair distance is further than maxDistSq
MRMESH_API size_t deactivateFarPairs( IPointPairs& pairs, float maxDistSq );

/// in each pair updates the target data and performs basic filtering (activation)
MRMESH_API void updatePointPairs( PointPairs& pairs,
    const MeshOrPointsXf& src, const MeshOrPointsXf& tgt,
    float cosThreshold, float distThresholdSq, bool mutualClosest );

/// This class allows you to register two object with similar shape using
/// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
/// \snippet cpp-examples/MeshICP.dox.cpp 0
class [[nodiscard]] ICP
{
public:
    /// Constructs ICP framework with given sample points on both objects
    /// \param flt floating object
    /// \param ref reference object
    /// \param fltXf transformation from floating object space to global space
    /// \param refXf transformation from reference object space to global space
    /// \param fltSamples samples on floating object to find projections on the reference object during the algorithm
    /// \param refSamples samples on reference object to find projections on the floating object during the algorithm
    ICP( const MeshOrPoints& flt, const MeshOrPoints& ref, const AffineXf3f& fltXf, const AffineXf3f& refXf,
        const VertBitSet& fltSamples = {}, const VertBitSet& refSamples = {} ) : ICP( { flt, fltXf }, { ref, refXf }, fltSamples, refSamples ) {}
    MRMESH_API ICP( const MeshOrPointsXf& flt, const MeshOrPointsXf& ref, const VertBitSet& fltSamples = {}, const VertBitSet& refSamples = {} );

    /// Constructs ICP framework with automatic points sampling on both objects
    /// \param flt floating object
    /// \param ref reference object
    /// \param fltXf transformation from floating object space to global space
    /// \param refXf transformation from reference object space to global space
    /// \param samplingVoxelSize approximate distance between samples on each of two objects
    MRMESH_API ICP( const MeshOrPoints& flt, const MeshOrPoints& ref, const AffineXf3f& fltXf, const AffineXf3f& refXf,
        float samplingVoxelSize ) : ICP( { flt, fltXf }, { ref, refXf }, samplingVoxelSize ) {}
    MRMESH_API ICP( const MeshOrPointsXf& flt, const MeshOrPointsXf& ref, float samplingVoxelSize );

    /// tune algorithm params before run calculateTransformation()
    void setParams(const ICPProperties& prop) { prop_ = prop; }
    MRMESH_API void setCosineLimit(const float cos);
    MRMESH_API void setDistanceLimit( const float dist );
    MRMESH_API void setBadIterCount( const int iter );
    MRMESH_API void setFarDistFactor(const float factor);

    /// select pairs with origin samples on floating object
    MRMESH_API void setFltSamples( const VertBitSet& fltSamples );
    MRMESH_API void sampleFltPoints( float samplingVoxelSize );

    /// select pairs with origin samples on reference object
    MRMESH_API void setRefSamples( const VertBitSet& refSamples );
    MRMESH_API void sampleRefPoints( float samplingVoxelSize );

    /// select pairs with origin samples on both objects
    void samplePoints( float samplingVoxelSize ) { sampleFltPoints( samplingVoxelSize ); sampleRefPoints( samplingVoxelSize ); }

    [[deprecated]] MR_BIND_IGNORE void recomputeBitSet( float fltSamplingVoxelSize ) { sampleFltPoints( fltSamplingVoxelSize ); }

    /// sets to-world transformations both for floating and reference objects
    MRMESH_API void setXfs( const AffineXf3f& fltXf, const AffineXf3f& refXf );

    /// sets to-world transformation for the floating object
    MRMESH_API void setFloatXf( const AffineXf3f& fltXf );

    /// automatically selects initial transformation for the floating object
    /// based on covariance matrices of both floating and reference objects;
    /// applies the transformation to the floating object and returns it
    MRMESH_API AffineXf3f autoSelectFloatXf();

    /// recompute point pairs after manual change of transformations or parameters
    MRMESH_API void updatePointPairs();

    [[nodiscard]] const ICPProperties& getParams() const { return prop_; }

    [[nodiscard]] MRMESH_API std::string getStatusInfo() const; // returns status info string

    /// computes the number of samples able to form pairs
    [[nodiscard]] size_t getNumSamples() const { return MR::getNumSamples( flt2refPairs_ ) + MR::getNumSamples( ref2fltPairs_ ); }

    /// computes the number of active point pairs
    [[nodiscard]] size_t getNumActivePairs() const { return MR::getNumActivePairs( flt2refPairs_ ) + MR::getNumActivePairs( ref2fltPairs_ ); }

    /// computes root-mean-square deviation between points
    [[nodiscard]] MRMESH_API float getMeanSqDistToPoint() const;

    /// computes root-mean-square deviation from points to target planes
    [[nodiscard]] MRMESH_API float getMeanSqDistToPlane() const;

    /// returns current pairs formed from samples on floating object and projections on reference object
    [[nodiscard]] const PointPairs & getFlt2RefPairs() const { return flt2refPairs_; }

    /// returns current pairs formed from samples on reference object and projections on floating object
    [[nodiscard]] const PointPairs & getRef2FltPairs() const { return ref2fltPairs_; }

    /// runs ICP algorithm given input objects, transformations, and parameters;
    /// \return adjusted transformation of the floating object to match reference object
    [[nodiscard]] MRMESH_API AffineXf3f calculateTransformation();

private:
    MeshOrPointsXf flt_;
    MeshOrPointsXf ref_;

    ICPProperties prop_;

    PointPairs flt2refPairs_;
    PointPairs ref2fltPairs_;

    ICPExitType resultType_{ ICPExitType::NotStarted };

    /// deactivate pairs that does not meet farDistFactor criterion
    void deactivatefarDistPairs_();

    int iter_ = 0;
    bool p2ptIter_();
    bool p2plIter_();

    void calcGen_( float (ICP::*dist)() const, bool (ICP::*iter)() );
    void calcP2Pt_();
    void calcP2Pl_();
    void calcCombined_();
};

} //namespace MR
