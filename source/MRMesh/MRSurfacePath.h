#pragma once

#include "MRMeshFwd.h"
#include <vector>
#include <tl/expected.hpp>

namespace MR
{

/// \defgroup SurfacePathSubgroup Surface Path
/// \ingroup SurfacePathGroup
/// \{

enum class PathError
{
    StartEndNotConnected, ///< no path can be found from start to end, because they are not from the same connected component
    InternalError         ///< report to developers for investigation
};

/// returns intermediate points of the geodesic path from start to end, where it crosses mesh edges;
/// the path can be limited to given region: in face-format inside mp, or in vert-format in vertRegion argument.
/// It is the same as calling computeSurfacePathApprox() then reducePath()
MRMESH_API tl::expected<SurfacePath, PathError> computeSurfacePath( const MeshPart & mp, 
    const MeshTriPoint & start, const MeshTriPoint & end, int numPostProcessIters = 5, const VertBitSet* vertRegion = nullptr,
    Vector<float, VertId> * outSurfaceDistances = nullptr );

/// returns intermediate points of approximately geodesic path from start to end, where it crosses mesh edges;
/// the path can be limited to given region: in face-format inside mp, or in vert-format in vertRegion argument
MRMESH_API tl::expected<SurfacePath, PathError> computeSurfacePathApprox( const MeshPart & mp, 
    const MeshTriPoint & start, const MeshTriPoint & end, const VertBitSet* vertRegion = nullptr,
    Vector<float, VertId> * outSurfaceDistances = nullptr );

/// for each vertex from (starts) finds the closest vertex from (ends) in geodesic sense
/// \param vertRegion consider paths going in this region only
MRMESH_API HashMap<VertId, VertId> computeClosestSurfacePathTargets( const Mesh & mesh,
    const VertBitSet & starts, const VertBitSet & ends, const VertBitSet * vertRegion = nullptr,
    Vector<float, VertId> * outSurfaceDistances = nullptr );

/// returns length of surface path, accumulate each segment
MRMESH_API float surfacePathLength( const Mesh& mesh, const SurfacePath& surfacePath );

/// \}

} // namespace MR
