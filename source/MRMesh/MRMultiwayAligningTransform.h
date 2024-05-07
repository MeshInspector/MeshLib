#pragma once

#include "MRMeshFwd.h"
#include "MRRigidXf3.h"
#include <MRPch/MREigenCore.h>

namespace MR
{

/// \defgroup AligningTransformGroup Aligning Transform
/// \ingroup MathGroup
/// \{

/// This class can be used to solve the problem of multiple 3D objects alignment.
class MultiwayAligningTransform
{
public:
    /// initializes internal data to start registering given number of objects
    void reset( int numObjs );

    /// appends a point-to-point link into consideration with given weight (w):
    /// one point (pA) from (objA), and the other point (pB) from (objB)
    //MRMESH_API void add( int objA, const Vector3d& pA, int objB, const Vector3d& pB, double w = 1 );

    /// appends a point-to-point link into consideration with given weight (w):
    /// one point (pA) from (objA), and the other point (pB) from (objB)
    //void add( int objA, const Vector3f& pA, int objB, const Vector3f& pB, float w = 1 ) { add( objA, Vector3d( pA ), objB, Vector3d( pB ), w ); }

    /// appends a point-to-line link into consideration with given weight (w):
    /// one point (pA) from (objA), and the other plane specified by point (pB) and normal (nB) from (objB)
    MRMESH_API void add( int objA, const Vector3d& pA, int objB, const Vector3d& pB, const Vector3d& nB, double w = 1 );

    /// appends a point-to-line link into consideration with given weight (w):
    /// one point (pA) from (objA), and the other plane specified by point (pB) and normal (nB) from (objB)
    void add( int objA, const Vector3f& pA, int objB, const Vector3f& pB, const Vector3f& nB, float w = 1 ) { add( objA, Vector3d( pA ), objB, Vector3d( pB ), Vector3d( nB ), w ); }

    /// finds the solution consisting of all objects transformations (numObj),
    /// that minimizes the summed weighted squared distance among accumulated links (point-point or point-plane);
    /// the transform of the last object is always identity (it is fixed)
    [[nodiscard]] MRMESH_API std::vector<RigidXf3d> solve();

private:
    Eigen::MatrixXd a_; ///< matrix of linear system to solve, only upper-right part is filled
    Eigen::VectorXd b_; ///< right hand size of the linear system to solve
    int numObjs_ = 0;
};

/// \}

} //namespace MR
