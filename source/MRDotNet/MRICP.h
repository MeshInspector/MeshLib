#pragma once
#include "MRMeshFwd.h"
#include "MRMesh/MRICPEnums.h"
#include "MRVector3.h"
#pragma managed( push, off )
#include <MRMesh/MRICP.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

/// Stores a pair of points: one samples on the source and the closest to it on the target
public ref struct PointPair
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

    /// id of the source point
    VertId srcVertId;

    /// for point clouds it is the closest vertex on target,
    /// for meshes it is the closest vertex of the triangle with the closest point on target
    VertId tgtCloseVert;

    /// cosine between normals in source and target points
    float normalsAngleCos = 1.f;

    /// true if if the closest point on target is located on the boundary (only for meshes)
    bool tgtOnBd = false;
};

public ref struct PointPairs
{
    PointPairs( const MR::PointPairs& pairs );

    List<PointPair^>^ pairs;
    BitSet^ active;
};

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
    Vector3f^ fixedRotationAxis = gcnew Vector3f( 0, 0, 0 );

    /// maximum iterations
    int iterLimit = 10;

    /// maximum iterations without improvements
    int badIterStopCount = 3;

    /// Algorithm target root-mean-square distance. As soon as it is reached, the algorithm stops.
    float exitVal = 0; // [distance]

    /// a pair of points is formed only if both points in the pair are mutually closest (reciprocity test passed)
    bool mutualClosest = false;

    MR::ICPProperties ToNative();
};

public ref class ICP
{
public:
    /// Constructs ICP framework with given sample points on both objects
    /// \param flt floating object and transformation from floating object space to global space
    /// \param ref reference object and transformation from reference object space to global space
    /// \param samplingVoxelSize approximate distance between samples on each of two objects
    ICP( MeshOrPointsXf^ flt, MeshOrPointsXf^ ref, float samplingVoxelSize );

    /// Constructs ICP framework with given sample points on both objects
    /// \param flt floating object and transformation from floating object space to global space
    /// \param ref reference object and transformation from reference object space to global space
    /// \param fltSamples samples on floating object to find projections on the reference object during the algorithm
    /// \param refSamples samples on reference object to find projections on the floating object during the algorithm
    ICP( MeshOrPointsXf^ flt, MeshOrPointsXf^ ref, BitSet^ fltSamples, BitSet^ refSamples );
    ~ICP();
    /// tune algorithm params before run calculateTransformation()
    void SetParams( ICPProperties^ props );
    /// select pairs with origin samples on both objects
    void SamplePoints( float sampleVoxelSize );
    /// automatically selects initial transformation for the floating object
    /// based on covariance matrices of both floating and reference objects;
    /// applies the transformation to the floating object and returns it
    void AutoSelectFloatXf();
    /// recompute point pairs after manual change of transformations or parameters
    void UpdatePointPairs();
    /// returns status info string
    System::String^ GetStatusInfo();
    /// computes the number of samples able to form pairs
    int GetNumSamples();
    /// computes the number of active point pairs
    int GetNumActivePairs();
    /// computes root-mean-square deviation between points
    float GetMeanSqDistToPoint();
    /// computes root-mean-square deviation from points to target planes
    float GetMeanSqDistToPlane();
    /// returns current pairs formed from samples on floating object and projections on reference object
    PointPairs^ GetFlt2RefPairs();
    /// returns current pairs formed from samples on reference object and projections on floating object
    PointPairs^ GetRef2FltPairs();
    /// runs ICP algorithm given input objects, transformations, and parameters;
    /// \return adjusted transformation of the floating object to match reference object
    AffineXf3f^ CalculateTransformation();

private:
    MR::ICP* icp_;
};

MR_DOTNET_NAMESPACE_END

