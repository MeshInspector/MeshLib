#pragma once

#include "MRMeshFwd.h"
#include "MRBox.h"
#include "MRVector3.h"

namespace MR
{

using FoundPointCallback = std::function<void( VertId, const Vector3f& )>;

/// Finds all valid points of pointCloud that are inside or on the surface of given box
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
MRMESH_API void findPointsInBox( const PointCloud& pointCloud, const Box3f& box,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr );

/// Finds all valid vertices of the mesh that are inside or on the surface of given box
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
MRMESH_API void findPointsInBox( const Mesh& mesh, const Box3f& box,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr );

/// Finds all points in tree that are inside or on the surface of given box
/// \ingroup AABBTreeGroup
/// \param xf points-to-center transformation, if not specified then identity transformation is assumed
MRMESH_API void findPointsInBox( const AABBTreePoints& tree, const Box3f& box,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf = nullptr );

} //namespace MR
