#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRId.h"
#include "MRMeshOrPoints.h"
#include "MRVector3.h"

MR_EXTERN_C_BEGIN

/// The method how to update transformation from point pairs
typedef enum MRICPMethod
{
    /// PointToPoint for the first 2 iterations, and PointToPlane for the remaining iterations
    MRICPMethodCombined = 0,
    /// select transformation that minimizes mean squared distance between two points in each pair,
    /// it is the safest approach but can converge slowly
    MRICPMethodPointToPoint = 1,
    /// select transformation that minimizes mean squared distance between a point and a plane via the other point in each pair,
    /// converge much faster than PointToPoint in case of many good (with not all points/normals in one plane) pairs
    MRICPMethodPointToPlane = 2
} MRICPMethod;

/// The group of transformations, each with its own degrees of freedom
typedef enum MRICPMode
{
    /// rigid body transformation with uniform scaling (7 degrees of freedom)
    MRICPModeRigidScale = 0,
    /// rigid body transformation (6 degrees of freedom)
    MRICPModeAnyRigidXf,
    /// rigid body transformation with rotation except argument axis (5 degrees of freedom)
    MRICPModeOrthogonalAxis,
    /// rigid body transformation with rotation around given axis only (4 degrees of freedom)
    MRICPModeFixedAxis,
    /// only translation (3 degrees of freedom)
    MRICPModeTranslationOnly
} MRICPMode;

typedef struct MRICPPairData
{
    /// coordinates of the source point after transforming in world space
    MRVector3f srcPoint;
    /// normal in source point after transforming in world space
    MRVector3f srcNorm;
    /// coordinates of the closest point on target after transforming in world space
    MRVector3f tgtPoint;
    /// normal in the target point after transforming in world space
    MRVector3f tgtNorm;
    /// squared distance between source and target points
    float distSq;
    /// weight of the pair (to prioritize over other pairs)
    float weight;
} MRICPPairData;

/// Stores a pair of points: one samples on the source and the closest to it on the target
typedef struct MRPointPair
{
    MRICPPairData ICPPairData;
    /// id of the source point
    MRVertId srcVertId;
    /// for point clouds it is the closest vertex on target,
    /// for meshes it is the closest vertex of the triangle with the closest point on target
    MRVertId tgtCloseVert;
    /// cosine between normals in source and target points
    float normalsAngleCos;
    /// true if if the closest point on target is located on the boundary (only for meshes)
    bool tgtOnBd;
} MRPointPair;

/// Simple interface for pairs holder
typedef struct MRIPointPairs MRIPointPairs;

typedef struct MRPointPairs MRPointPairs;

MRMESHC_API const MRICPPairData* mrIPointPairsGet( const MRIPointPairs* pp, size_t idx );

MRMESHC_API size_t mrIPointPairsSize( const MRIPointPairs* pp );

MRMESHC_API MRICPPairData* mrIPointPairsGetRef( MRIPointPairs* pp, size_t idx );

/// types of exit conditions in calculation
typedef enum MRICPExitType
{
    /// calculation is not started yet
    MRICPExitTypeNotStarted = 0,
    /// solution not found in some iteration
    MRICPExitTypeNotFoundSolution,
    /// iteration limit reached
    MRICPExitTypeMaxIterations,
    /// limit of non-improvement iterations in a row reached
    MRICPExitTypeMaxBadIterations,
    /// stop mean square deviation reached
    MRICPExitTypeStopMsdReached
} MRICPExitType;

typedef struct MRICPProperties
{
    /// The method how to update transformation from point pairs
    MRICPMethod method;
    /// Rotation angle during one iteration of PointToPlane will be limited by this value
    float p2plAngleLimit;
    /// Scaling during one iteration of PointToPlane will be limited by this value
    float p2plScaleLimit;
    /// Points pair will be counted only if cosine between surface normals in points is higher
    float cosThreshold;
    /// Points pair will be counted only if squared distance between points is lower than
    float distThresholdSq;
    /// Points pair will be counted only if distance between points is lower than
    /// root-mean-square distance times this factor
    float farDistFactor;
    /// Finds only translation. Rotation part is identity matrix
    MRICPMode icpMode;
    /// If this vector is not zero then rotation is allowed relative to this axis only
    MRVector3f fixedRotationAxis;
    /// maximum iterations
    int iterLimit;
    /// maximum iterations without improvements
    int badIterStopCount;
    /// Algorithm target root-mean-square distance. As soon as it is reached, the algorithm stops.
    float exitVal;
    /// a pair of points is formed only if both points in the pair are mutually closest (reciprocity test passed)
    bool mutualClosest;
} MRICPProperties;

/// initializes a default instance
MRMESHC_API MRICPProperties mrICPPropertiesNew( void );

/// This class allows you to register two object with similar shape using
/// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
typedef struct MRICP MRICP;

/// Constructs ICP framework with automatic points sampling on both objects
MRMESHC_API MRICP* mrICPNew( const MRMeshOrPointsXf* flt, const MRMeshOrPointsXf* ref, float samplingVoxelSize );
/// Constructs ICP framework with given sample points on both objects
MRMESHC_API MRICP* mrICPNewFromSamples( const MRMeshOrPointsXf* flt, const MRMeshOrPointsXf* ref, const MRVertBitSet* fltSamples, const MRVertBitSet* refSamples );

/// tune algorithm params before run calculateTransformation()
MRMESHC_API void mrICPSetParams( MRICP* icp, const MRICPProperties* prop );

/// select pairs with origin samples on both objects
MRMESHC_API void mrICPSamplePoints( MRICP* icp, float samplingVoxelSize );

/// automatically selects initial transformation for the floating object
/// based on covariance matrices of both floating and reference objects;
/// applies the transformation to the floating object and returns it
MRMESHC_API MRAffineXf3f mrICPAutoSelectFloatXf( MRICP* icp );

/// recompute point pairs after manual change of transformations or parameters
MRMESHC_API void mrICPUpdatePointPairs( MRICP* icp );

/// returns status info string
MRMESHC_API MRString* mrICPGetStatusInfo( const MRICP* icp );

/// computes the number of samples able to form pairs
MRMESHC_API size_t mrICPGetNumSamples( const MRICP* icp );

/// computes the number of active point pairs
MRMESHC_API size_t mrICPGetNumActivePairs( const MRICP* icp );

/// computes root-mean-square deviation between points
MRMESHC_API float mrICPGetMeanSqDistToPoint( const MRICP* icp );

/// computes root-mean-square deviation from points to target planes
MRMESHC_API float mrICPGetMeanSqDistToPlane( const MRICP* icp );

/// returns current pairs formed from samples on floating object and projections on reference object
MRMESHC_API const MRIPointPairs* mrICPGetFlt2RefPairs( const MRICP* icp );

/// returns current pairs formed from samples on reference object and projections on floating object
MRMESHC_API const MRIPointPairs* mrICPGetRef2FltPairs( const MRICP* icp );

/// runs ICP algorithm given input objects, transformations, and parameters;
/// \return adjusted transformation of the floating object to match reference object
MRMESHC_API MRAffineXf3f mrICPCalculateTransformation( MRICP* icp );

/// deallocates an ICP object
MRMESHC_API void mrICPFree( MRICP* icp );

MR_EXTERN_C_END
