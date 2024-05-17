#pragma once

#include "MRMeshFwd.h"
#include "MRRigidXf3.h"
#include <memory>

namespace MR
{

/// \defgroup AligningTransformGroup Aligning Transform
/// \ingroup MathGroup
/// \{

/// This class can be used to solve the problem of multiple 3D objects alignment,
/// by first collecting weighted links between pairs of points from different objects,
/// and then solving for transformations minimizing weighted average of link penalties
class MultiwayAligningTransform
{
public:
    /// initializes internal data to start registering given number of objects
    MRMESH_API explicit MultiwayAligningTransform( int numObjs = 0 );

    MRMESH_API ~MultiwayAligningTransform();

    /// reinitializes internal data to start registering given number of objects
    void reset( int numObjs );

    /// appends a 3D link into consideration: one point (pA) from (objA), and the other point (pB) from (objB)
    /// with link penalty equal to weight (w) times squared distance between two points
    MRMESH_API void add( int objA, const Vector3d& pA, int objB, const Vector3d& pB, double w = 1 );

    /// appends a 3D link into consideration: one point (pA) from (objA), and the other point (pB) from (objB)
    /// with link penalty equal to weight (w) times squared distance between two points
    void add( int objA, const Vector3f& pA, int objB, const Vector3f& pB, float w = 1 ) { add( objA, Vector3d( pA ), objB, Vector3d( pB ), w ); }

    /// appends a 1D link into consideration: one point (pA) from (objA), and the other point (pB) from (objB)
    /// with link penalty equal to weight (w) times squared distance between their projections on given direction (n);
    /// for a point on last fixed object, it is equivalent to point-to-plane link with the plane through that fixed point with normal (n)
    MRMESH_API void add( int objA, const Vector3d& pA, int objB, const Vector3d& pB, const Vector3d& n, double w = 1 );

    /// appends a 1D link into consideration: one point (pA) from (objA), and the other point (pB) from (objB)
    /// with link penalty equal to weight (w) times squared distance between their projections on given direction (n);
    /// for a point on last fixed object, it is equivalent to point-to-plane link with the plane through that fixed point with normal (n)
    void add( int objA, const Vector3f& pA, int objB, const Vector3f& pB, const Vector3f& n, float w = 1 ) { add( objA, Vector3d( pA ), objB, Vector3d( pB ), Vector3d( n ), w ); }

    /// appends links accumulated in (r) into this
    MRMESH_API void add( const MultiwayAligningTransform & r );

    /// finds the solution consisting of all objects transformations (numObj),
    /// that minimizes the summed weighted squared distance among accumulated links;
    /// the transform of the last object is always identity (it is fixed)
    [[nodiscard]] MRMESH_API std::vector<RigidXf3d> solve();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// \}

} //namespace MR
