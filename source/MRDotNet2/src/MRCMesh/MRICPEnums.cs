public static partial class MR
{
    /// The method how to update transformation from point pairs
    public enum ICPMethod : int
    {
        ///< PointToPoint for the first 2 iterations, and PointToPlane for the remaining iterations
        Combined = 0,
        ///< select transformation that minimizes mean squared distance between two points in each pair,
        ///< it is the safest approach but can converge slowly
        PointToPoint = 1,
        ///< select transformation that minimizes mean squared distance between a point and a plane via the other point in each pair,
        ///< converge much faster than PointToPoint in case of many good (with not all points/normals in one plane) pairs
        ///< This mode only works correctly if the object that points are being projected onto has normals information (is not a point clouds without normals).
        ///< By default both objects are projected onto each other, so at least one of them must have normals, unless you change the settings
        ///< to only project one way (using `MR::ICP::setFltSamples()` or `MR::ICP::setRefSamples()`), in which case the object being projected onto
        ///< must have normals.
        PointToPlane = 2,
    }

    /// The group of transformations, each with its own degrees of freedom
    public enum ICPMode : int
    {
        ///< rigid body transformation with uniform scaling (7 degrees of freedom)
        RigidScale = 0,
        ///< rigid body transformation (6 degrees of freedom)
        AnyRigidXf = 1,
        ///< rigid body transformation with rotation except argument axis (5 degrees of freedom)
        OrthogonalAxis = 2,
        ///< rigid body transformation with rotation around given axis only (4 degrees of freedom)
        FixedAxis = 3,
        ///< only translation (3 degrees of freedom)
        TranslationOnly = 4,
    }

    // types of exit conditions in calculation
    public enum ICPExitType : int
    {
        // calculation is not started yet
        NotStarted = 0,
        // solution not found in some iteration
        NotFoundSolution = 1,
        // iteration limit reached
        MaxIterations = 2,
        // limit of non-improvement iterations in a row reached
        MaxBadIterations = 3,
        // stop mean square deviation reached
        StopMsdReached = 4,
    }
}
