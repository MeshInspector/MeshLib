#pragma once

#include "MRMeshOrPoints.h"
#include "MRMatrix3.h"
#include "MRId.h"
#include "MRConstants.h"
#include "MRAffineXf.h"
#include "MRBitSet.h"

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

    /// normal in source point after transforming in world space
    Vector3f srcNorm;

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

    friend bool operator == ( const PointPair&, const PointPair& ) = default;
};

struct PointPairs
{
    std::vector<PointPair> vec;
    BitSet active; ///< whether corresponding pair from vec must be considered during minimization
};

/// computes the number of active pairs
[[nodiscard]] MRMESH_API size_t getNumActivePairs( const PointPairs & pairs );

/// computes root-mean-square deviation between points
[[nodiscard]] MRMESH_API float getMeanSqDistToPoint( const PointPairs & pairs );

/// computes root-mean-square deviation from points to target planes
[[nodiscard]] MRMESH_API float getMeanSqDistToPlane( const PointPairs & pairs, const MeshOrPoints & floating, const AffineXf3f & floatXf );

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
};

/// This class allows you to register two object with similar shape using
/// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
class [[nodiscard]] ICP
{
public:
    /// xf parameters should represent current transformations of meshes
    /// fltXf transform from the local floating basis to the global
    /// refXf transform from the local reference basis to the global
    /// fltSamples allows to take exact set of vertices from the floating object
    MRMESH_API ICP(const MeshOrPoints& floating, const MeshOrPoints& reference, const AffineXf3f& fltXf, const AffineXf3f& refXf,
        const VertBitSet& fltSamples);

    MRMESH_API ICP(const MeshOrPoints& floating, const MeshOrPoints& reference, const AffineXf3f& fltXf, const AffineXf3f& refXf,
        float floatSamplingVoxelSize ); // positive value here defines voxel size, and only one vertex per voxel will be selected
    // TODO: add single transform constructor

    /// tune algorithm params before run calculateTransformation()
    void setParams(const ICPProperties& prop) { prop_ = prop; }
    MRMESH_API void setCosineLimit(const float cos);
    MRMESH_API void setDistanceLimit( const float dist );
    MRMESH_API void setBadIterCount( const int iter );
    MRMESH_API void setFarDistFactor(const float factor);
    MRMESH_API void recomputeBitSet(const float floatSamplingVoxelSize);

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
    [[nodiscard]] size_t getNumActivePairs() const { return MR::getNumActivePairs( flt2refPairs_ ); }

    /// computes root-mean-square deviation between points
    [[nodiscard]] float getMeanSqDistToPoint() const { return MR::getMeanSqDistToPoint( flt2refPairs_ ); }

    /// computes root-mean-square deviation from points to target planes
    [[nodiscard]] float getMeanSqDistToPlane() const { return MR::getMeanSqDistToPlane( flt2refPairs_, flt_, fltXf_ ); }

    /// returns current pairs formed from samples on floating and projections on reference
    [[nodiscard]] const PointPairs & getFlt2RefPairs() const { return flt2refPairs_; }

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
