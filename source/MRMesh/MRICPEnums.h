#pragma once
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
    ///< This mode only works correctly if the object that points are being projected onto has normals information (is not a point clouds without normals).
    ///< By default both objects are projected onto each other, so at least one of them must have normals, unless you change the settings
    ///< to only project one way (using `MR::ICP::setFltSamples()` or `MR::ICP::setRefSamples()`), in which case the object being projected onto
    ///< must have normals.
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

// types of exit conditions in calculation
enum class ICPExitType {
    NotStarted, // calculation is not started yet
    NotFoundSolution, // solution not found in some iteration
    MaxIterations, // iteration limit reached
    MaxBadIterations, // limit of non-improvement iterations in a row reached
    StopMsdReached // stop mean square deviation reached
};
}
