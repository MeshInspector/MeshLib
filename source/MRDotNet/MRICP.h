#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

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

public ref struct ICPPairData
{
    /// coordinates of the source point after transforming in world space
    Vector3f^ srcPoint;

    /// normal in source point after transforming in world space
    Vector3f^ srcNorm;

    /// coordinates of the closest point on target after transforming in world space
    Vector3f^ tgtPoint;

    /// normal in the target point after transforming in world space
    Vector3f^ tgtNorm;

    /// squared distance between source and target points
    float distSq = 0.f;

    /// weight of the pair (to prioritize over other pairs)
    float weight = 1.f;

    //static bool operator == ( ICPPairData^ a, ICPPairData^ b );
};

/// Stores a pair of points: one samples on the source and the closest to it on the target
public ref struct PointPair : public ICPPairData
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

   // static bool operator == ( PointPair^, PointPair^ );
};

public ref struct PointPairs
{
    PointPairs( const MR::PointPairs& pairs );

    List<PointPair^>^ pairs;
    BitSet^ active;
};

// types of exit conditions in calculation
enum class ICPExitType {
    NotStarted, // calculation is not started yet
    NotFoundSolution, // solution not found in some iteration
    MaxIterations, // iteration limit reached
    MaxBadIterations, // limit of non-improvement iterations in a row reached
    StopMsdReached // stop mean square deviation reached
};

/*public ref struct NumSum
{
    int num = 0;
    double sum = 0;

    static NumSum^ operator + ( NumSum^ a, NumSum^ b );

    float rootMeanSqF();
};*/

public ref struct ICPProperties
{
    /// The method how to update transformation from point pairs
    ICPMethod method = ICPMethod::PointToPlane;

    /// Rotation angle during one iteration of PointToPlane will be limited by this value
    float p2plAngleLimit = float( System::Math::PI ) / 6.0f; // [radians]

    /// Scaling during one iteration of PointToPlane will be limited by this value
    float p2plScaleLimit = 2;

    /// Points pair will be counted only if cosine between surface normals in points is higher
    float cosThreshold = 0.7f; // in [-1,1]

    /// Points pair will be counted only if squared distance between points is lower than
    float distThresholdSq = 1.f; // [distance^2]

    /// Points pair will be counted only if distance between points is lower than
    /// root-mean-square distance times this factor
    float farDistFactor = 3.f; // dimensionless

    /// Finds only translation. Rotation part is identity matrix
    ICPMode icpMode = ICPMode::AnyRigidXf;

    /// If this vector is not zero then rotation is allowed relative to this axis only
    Vector3f^ fixedRotationAxis;

    /// maximum iterations
    int iterLimit = 10;

    /// maximum iterations without improvements
    int badIterStopCount = 3;

    /// Algorithm target root-mean-square distance. As soon as it is reached, the algorithm stops.
    float exitVal = 0; // [distance]

    /// a pair of points is formed only if both points in the pair are mutually closest (reciprocity test passed)
    bool mutualClosest = false;
};

public ref class ICP
{
public:
    /*static int GetNumSamples(PointPairs^ pairs);
    static int GetNumActivePairs( PointPairs^ pairs );

    static NumSum^ GetSumSqDistToPoint( PointPairs^ pairs );
    static NumSum^ GetSumSqDistToPoint( PointPairs^ pairs, double inaccuracy );

    static NumSum^ GetSumSqDistToPlane( PointPairs^ pairs );
    static NumSum^ GetSumSqDistToPlane( PointPairs^ pairs, double inaccuracy );

    float GetMeanSqDistToPoint( PointPairs^ pairs );
    float GetMeanSqDistToPlane( PointPairs^ pairs );

    System::String^ GetICPStatusInfo( int iterations, ICPExitType exitType );

    /// given prepared (p2pl) object, finds the best transformation from it of given type with given limitations on rotation angle and global scale
    AffineXf3f^ GetAligningXf( const PointToPlaneAligningTransform& p2pl,
        ICPMode mode, float angleLimit, float scaleLimit, const Vector3f& fixedRotationAxis );*/

    ICP( MeshOrPointsXf^ flt, MeshOrPointsXf^ ref, float samplingVoxelSize );
    ICP( MeshOrPointsXf^ flt, MeshOrPointsXf^ ref, BitSet^ fltSamples, BitSet^ refSamples );
    ~ICP();

    void SetParams( ICPProperties^ props );
    void SamplePoints( float sampleVoxelSize );

    void AutoSelectFloatXf();
    void UpdatePointPairs();

    System::String^ GetStatusInfo();

    int GetNumSamples();
    int GetNumActivePairs();

    float GetMeanSqDistToPoint();
    float GetMeanSqDistToPlane();

    PointPairs^ GetFlt2RefPairs();
    PointPairs^ GetRef2FltPairs();

    AffineXf3f^ CalculateTransformation();

private:
    MR::ICP* icp_;
};

MR_DOTNET_NAMESPACE_END

