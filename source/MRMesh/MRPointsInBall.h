#pragma once

#include "MRMeshFwd.h"
#include "MRBall.h"
#include "MRVector3.h"

namespace MR
{

using FoundPointCallback = std::function<void( VertId, const Vector3f& )>;

/// Finds all valid points of pointCloud that are inside or on the surface of given ball
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
MRMESH_API void findPointsInBall( const PointCloud& pointCloud, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr );
[[deprecated]] inline void findPointsInBall( const PointCloud& pointCloud, const Vector3f& center, float radius,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr ) { return findPointsInBall( pointCloud, { center, sqr( radius ) }, foundCallback, xf ); }

/// Finds all valid vertices of the mesh that are inside or on the surface of given ball
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
MRMESH_API void findPointsInBall( const Mesh& mesh, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr );
[[deprecated]] inline void findPointsInBall( const Mesh& mesh, const Vector3f& center, float radius,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr ) { return findPointsInBall( mesh, { center, sqr( radius ) }, foundCallback, xf ); }

/// Finds all points in tree that are inside or on the surface of given ball
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
MRMESH_API void findPointsInBall( const AABBTreePoints& tree, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr );
[[deprecated]] inline void findPointsInBall( const AABBTreePoints& tree, const Vector3f& center, float radius,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr ) { return findPointsInBall( tree, { center, sqr( radius ) }, foundCallback, xf ); }

} //namespace MR
