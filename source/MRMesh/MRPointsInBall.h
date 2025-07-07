#pragma once

#include "MRPch/MRBindingMacros.h"
#include "MRMeshFwd.h"
#include "MRBall.h"
#include "MRVector3.h"
#include "MRPointsProject.h"
#include "MREnums.h"

namespace MR
{

using FoundPointCallback = std::function<void( VertId, const Vector3f& )>;

/// this callback is invoked on every point located within the ball, and allows changing the ball for search continuation
using OnPointInBallFound = std::function<Processing( const PointsProjectionResult & found, const Vector3f & foundXfPos, Ball3f & ball )>;

/// Finds all valid points of pointCloud that are inside or on the surface of given ball until callback returns Stop;
/// the ball can shrink (new ball is always within the previous one) during the search but never expand
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
MRMESH_API void findPointsInBall( const PointCloud& pointCloud, const Ball3f & ball,
    const OnPointInBallFound& foundCallback, const AffineXf3f* xf = nullptr );

/// Finds all valid points of pointCloud that are inside or on the surface of given ball
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
[[deprecated]] MRMESH_API MR_BIND_IGNORE void findPointsInBall( const PointCloud& pointCloud, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr );

/// Finds all valid vertices of the mesh that are inside or on the surface of given ball until callback returns Stop;
/// the ball can shrink (new ball is always within the previous one) during the search but never expand
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
MRMESH_API void findPointsInBall( const Mesh& mesh, const Ball3f& ball,
    const OnPointInBallFound& foundCallback, const AffineXf3f* xf = nullptr );

/// Finds all valid vertices of the mesh that are inside or on the surface of given ball
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
[[deprecated]] MRMESH_API MR_BIND_IGNORE void findPointsInBall( const Mesh& mesh, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr );

/// Finds all points in tree that are inside or on the surface of given ball until callback returns Stop;
/// the ball can shrink (new ball is always within the previous one) during the search but never expand
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
MRMESH_API void findPointsInBall( const AABBTreePoints& tree, Ball3f ball,
    const OnPointInBallFound& foundCallback, const AffineXf3f* xf = nullptr );

/// Finds all points in tree that are inside or on the surface of given ball
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
[[deprecated]] MRMESH_API MR_BIND_IGNORE void findPointsInBall( const AABBTreePoints& tree, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr );

} //namespace MR
