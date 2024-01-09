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
    PointToPoint = 1, ///< select transformation that minimizes mean squared distance between two points in each, it is the safest approach but can converge slowly
    PointToPlane = 2  ///< select transformation that minimizes mean squared distance between a point and a plane via the other point in each pair, converge much faster in case of many good pairs
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

struct VertPair
{
    // coordinates of the closest point on reference mesh (after applying refXf)
    Vector3f refPoint;
    // surface normal in a vertex on the floating mesh (after applying fltXf)
    Vector3f norm;
    // surface normal in a vertex on the reference mesh (after applying refXf)
    Vector3f normRef;
    // ID of the floating mesh vertex
    VertId vertId;
    // This is cosine between normals in first(floating mesh) and second(reference mesh) points
    // It evaluates how good is this pair
    float normalsAngleCos = 1.f;
    // Storing squared distance between vertices
    float vertDist2 = 0.f;
    // weight of the pair with respect to the sum of adjoining triangles square
    float weight = 1.f;

    friend bool operator == ( const VertPair&, const VertPair& ) = default;
};

using VertPairs = std::vector<VertPair>;

/// computes root-mean-square deviation between points
[[nodiscard]] MRMESH_API float getMeanSqDistToPoint( const VertPairs & pairs );

// computes root-mean-square deviation from points to target planes
[[nodiscard]] MRMESH_API float getMeanSqDistToPlane( const VertPairs & pairs, const MeshOrPoints & floating, const AffineXf3f & floatXf );

struct ICPProperties
{
    // The method how to update transformation from point pairs
    ICPMethod method = ICPMethod::PointToPoint;
    // Rotation angle during one iteration of PointToPlane will be limited by this value
    float p2plAngleLimit = PI_F / 6.0f; // [radians]
    // Scaling during one iteration of PointToPlane will be limited by this value
    float p2plScaleLimit = 2;
    // Points pair will be counted only if cosine between surface normals in points is higher
    float cosTreshold = 0.7f; // in [-1,1]
    // Points pair will be counted only if squared distance between points is lower than
    float distTresholdSq = 1.f; // [distance^2]
    // Sigma multiplier for statistic throw of paints pair based on the distance
    // Default: all pairs in the interval the (distance = mean +- 3*sigma) are passed
    float distStatisticSigmaFactor = 3.f; // dimensionless
    // Finds only translation. Rotation part is identity matrix
    ICPMode icpMode = ICPMode::AnyRigidXf;
    // If this vector is not zero then rotation is allowed relative to this axis only
    Vector3f fixedRotationAxis;
    // keep point pairs from first iteration
    bool freezePairs = false;

    // parameters of iterative call
    int iterLimit = 30; // maximum iterations
    int badIterStopCount = 3; // maximum iterations without improvements

    // Algorithm target root-mean-square distance. As soon as it is reached, the algorithm stops.
    float exitVal = 0; // [distance]
};

/// This class allows you to register two object with similar shape using
/// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
class ICP
{
public:
    /// xf parameters should represent current transformations of meshes
    /// fltXf transform from the local floating basis to the global
    /// refXf transform from the local reference basis to the global
    /// floatBitSet allows to take exact set of vertices from the floating object
    MRMESH_API ICP(const MeshOrPoints& floating, const MeshOrPoints& reference, const AffineXf3f& fltXf, const AffineXf3f& refXf,
        const VertBitSet& floatBitSet);
    MRMESH_API ICP(const MeshOrPoints& floating, const MeshOrPoints& reference, const AffineXf3f& fltXf, const AffineXf3f& refXf,
        float floatSamplingVoxelSize ); // positive value here defines voxel size, and only one vertex per voxel will be selected
    // TODO: add single transform constructor

    /// tune algirithm params before run calculateTransformation()
    void setParams(const ICPProperties& prop) { prop_ = prop; }
    MRMESH_API void setCosineLimit(const float cos);
    MRMESH_API void setDistanceLimit( const float dist );
    MRMESH_API void setBadIterCount( const int iter );
    MRMESH_API void setPairsWeight(const std::vector<float> w);
    MRMESH_API void setDistanceFilterSigmaFactor(const float factor);
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
    MRMESH_API void updateVertPairs();

    const ICPProperties& getParams() const { return prop_; }
    MRMESH_API Vector3f getShiftVector() const; // shows mean pair vector
    MRMESH_API std::string getLastICPInfo() const; // returns status info string

    /// computes root-mean-square deviation between points
    float getMeanSqDistToPoint() const { return MR::getMeanSqDistToPoint( vertPairs_ ); }

    /// computes root-mean-square deviation from points to target planes
    float getMeanSqDistToPlane() const { return MR::getMeanSqDistToPlane( vertPairs_, floating_, floatXf_ ); }

    /// used to visualize generated points pairs
    const std::vector<VertPair>& getVertPairs() const { return vertPairs_; }

    /// finds squared minimum and maximum pairs distances
    MRMESH_API std::pair<float, float> getDistLimitsSq() const;

    /// returns new xf transformation for the floating mesh, which allows to match reference mesh
    MRMESH_API AffineXf3f calculateTransformation();

private:
    MeshOrPoints floating_;
    AffineXf3f floatXf_;
    VertBitSet floatVerts_; ///< vertices of floating object to find their pairs on reference mesh
    
    MeshOrPoints ref_;
    AffineXf3f refXf_;

    AffineXf3f float2refXf_; ///< transformation from floating object space to reference object space

    ICPProperties prop_;

    std::vector<VertPair> vertPairs_;

    // types of exit conditions in calculation
    enum class ExitType {
        NotStarted, // calculation is not started yet
        NotFoundSolution, // solution not found in some iteration
        MaxIterations, // iteration limit reached
        MaxBadIterations, // limit of non-improvement iterations in a row reached
        StopMsdReached // stop mean square deviation reached
    };
    ExitType resultType_{ ExitType::NotStarted };

    void removeInvalidVertPairs_();

    void updateVertFilters_();

    int iter_ = 0;
    bool p2ptIter_();
    bool p2plIter_();
};

using MeshICP [[deprecated]] = ICP;

} //namespace MR
