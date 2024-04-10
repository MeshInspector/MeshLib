#pragma once

#include "MRMeshOrPoints.h"
#include "MRMatrix3.h"
#include "MRId.h"
#include "MRConstants.h"
#include "MRAffineXf.h"
#include "MRBitSet.h"
#include <cfloat>

namespace MR
{

/// The method how to update transformation from point pairs
enum class ICPMethod
{
    Combined = 0,     ///< PointToPoint for the first 2 iterations, and PointToPlane for the remaining iterations
    PointToPoint = 1, ///< select transformation that minimizes mean squared distance between two points in each pair,
                      ///< it is the safest approach but can converge slowly
    PointToPlane = 2  ///< select transformation that minimizes mean squared distance between a point and a plane via the other point in each pair,
                      ///< converge much faster than PointToPoint in case of many good (with not all points/normals in one plane) pairs
};

/// The group of transformations, each with its own degrees of freedom
enum class ICPMode
{
    RigidScale,     ///< rigid body transformation with uniform scaling (7 degrees of freedom)
    AnyRigidXf,     ///< rigid body transformation (6 degrees of freedom)
    OrthogonalAxis, ///< rigid body transformation with rotation except argument axis (5 degrees of freedom)
    FixedAxis,      ///< rigid body transformation with rotation around given axis only (4 degrees of freedom)
    TranslationOnly ///< only translation (3 degrees of freedom)
};

/// Stores a pair of points: one samples on the source and the closest to it on the target
struct PointPair
{
    /// id of the source point
    VertId srcVertId;

    /// coordinates of the source point after transforming in world space
    Vector3f srcPoint;

    /// normal in source point after transforming in world space
    Vector3f srcNorm;

    /// for point clouds it is the closest vertex on target,
    /// for meshes it is the closest vertex of the triangle with the closest point on target
    VertId tgtCloseVert;

    /// coordinates of the closest point on target after transforming in world space
    Vector3f tgtPoint;

    /// normal in the target point after transforming in world space
    Vector3f tgtNorm;

    /// cosine between normals in source and target points
    float normalsAngleCos = 1.f;

    /// squared distance between source and target points
    float distSq = 0.f;

    /// weight of the pair (to prioritize over other pairs)
    float weight = 1.f;

    /// true if if the closest point on target is located on the boundary (only for meshes)
    bool tgtOnBd = false;

    friend bool operator == ( const PointPair&, const PointPair& ) = default;
};

struct PointPairs
{
    std::vector<PointPair> vec;
    BitSet active; ///< whether corresponding pair from vec must be considered during minimization
};

/// computes the number of active pairs
[[nodiscard]] MRMESH_API size_t getNumActivePairs( const PointPairs & pairs );

struct NumSum
{
    int num = 0;
    double sum = 0;

    friend NumSum operator + ( const NumSum & a, const NumSum & b ) { return { a.num + b.num, a.sum + b.sum }; }

    [[nodiscard]] float rootMeanSqF() const { return ( num <= 0 ) ? FLT_MAX : (float)std::sqrt( sum / num ); }
};

/// computes the number of active pairs and the sum of squared distances between points
[[nodiscard]] MRMESH_API NumSum getSumSqDistToPoint( const PointPairs & pairs );

/// computes the number of active pairs and the sum of squared deviation from points to target planes
[[nodiscard]] MRMESH_API NumSum getSumSqDistToPlane( const PointPairs & pairs );

/// computes root-mean-square deviation between points
[[nodiscard]] inline float getMeanSqDistToPoint( const PointPairs & pairs ) { return getSumSqDistToPoint( pairs ).rootMeanSqF(); }

/// computes root-mean-square deviation from points to target planes
[[nodiscard]] inline float getMeanSqDistToPlane( const PointPairs & pairs ) { return getSumSqDistToPlane( pairs ).rootMeanSqF(); }

struct ICPProperties
{
    /// The method how to update transformation from point pairs
    ICPMethod method = ICPMethod::PointToPlane;

    /// Rotation angle during one iteration of PointToPlane will be limited by this value
    float p2plAngleLimit = PI_F / 6.0f; // [radians]

    /// Scaling during one iteration of PointToPlane will be limited by this value
    float p2plScaleLimit = 2;

    /// Points pair will be counted only if cosine between surface normals in points is higher
    float cosTreshold = 0.7f; // in [-1,1]

    /// Points pair will be counted only if squared distance between points is lower than
    float distThresholdSq = 1.f; // [distance^2]

    /// Points pair will be counted only if distance between points is lower than
    /// root-mean-square distance times this factor
    float farDistFactor = 3.f; // dimensionless

    /// Finds only translation. Rotation part is identity matrix
    ICPMode icpMode = ICPMode::AnyRigidXf;

    /// If this vector is not zero then rotation is allowed relative to this axis only
    Vector3f fixedRotationAxis;

    /// maximum iterations
    int iterLimit = 10;

    /// maximum iterations without improvements
    int badIterStopCount = 3;

    /// Algorithm target root-mean-square distance. As soon as it is reached, the algorithm stops.
    float exitVal = 0; // [distance]

    /// a pair of points is formed only if both points in the pair are mutually closest (reciprocity test passed)
    bool mutualClosest = false;
};

/// This class allows you to register two object with similar shape using
/// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
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
    MRMESH_API ICP( const MeshOrPoints& flt, const MeshOrPoints& ref, const AffineXf3f& fltXf, const AffineXf3f& refXf,
        const VertBitSet& fltSamples = {}, const VertBitSet& refSamples = {} );

    /// Constructs ICP framework with automatic points sampling on both objects
    /// \param flt floating object
    /// \param ref reference object
    /// \param fltXf transformation from floating object space to global space
    /// \param refXf transformation from reference object space to global space
    /// \param samplingVoxelSize approximate distance between samples on each of two objects
    MRMESH_API ICP( const MeshOrPoints& flt, const MeshOrPoints& ref, const AffineXf3f& fltXf, const AffineXf3f& refXf,
        float samplingVoxelSize );

    /// tune algorithm params before run calculateTransformation()
    void setParams(const ICPProperties& prop) { prop_ = prop; }
    MRMESH_API void setCosineLimit(const float cos);
    MRMESH_API void setDistanceLimit( const float dist );
    MRMESH_API void setBadIterCount( const int iter );
    MRMESH_API void setFarDistFactor(const float factor);

    /// select pairs with origin samples on floating object
    MRMESH_API void sampleFltPoints( float samplingVoxelSize );

    /// select pairs with origin samples on reference object
    MRMESH_API void sampleRefPoints( float samplingVoxelSize );

    /// select pairs with origin samples on both objects
    void samplePoints( float samplingVoxelSize ) { sampleFltPoints( samplingVoxelSize ); sampleRefPoints( samplingVoxelSize ); }

    [[deprecated]] void recomputeBitSet( float fltSamplingVoxelSize ) { sampleFltPoints( fltSamplingVoxelSize ); }

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

    [[nodiscard]] MRMESH_API std::string getLastICPInfo() const; // returns status info string

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
    MeshOrPoints flt_;
    AffineXf3f fltXf_;
    
    MeshOrPoints ref_;
    AffineXf3f refXf_;

    ICPProperties prop_;

    PointPairs flt2refPairs_;
    PointPairs ref2fltPairs_;

    // types of exit conditions in calculation
    enum class ExitType {
        NotStarted, // calculation is not started yet
        NotFoundSolution, // solution not found in some iteration
        MaxIterations, // iteration limit reached
        MaxBadIterations, // limit of non-improvement iterations in a row reached
        StopMsdReached // stop mean square deviation reached
    };
    ExitType resultType_{ ExitType::NotStarted };

    /// in each pair updates the target data and performs basic filtering (activation)
    void updatePointPairs_( PointPairs & pairs,
        const MeshOrPoints & src, const AffineXf3f & srcXf,
        const MeshOrPoints & tgt, const AffineXf3f & tgtXf );

    /// deactivate pairs that does not meet farDistFactor criterion
    void deactivatefarDistPairs_();

    int iter_ = 0;
    bool p2ptIter_();
    bool p2plIter_();
};

} //namespace MR
