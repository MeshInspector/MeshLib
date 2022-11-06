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
/// It is the same as calling computeFastMarchingPath() then reducePath()
MRMESH_API tl::expected<SurfacePath, PathError> computeSurfacePath( const MeshPart & mp, 
    const MeshTriPoint & start, const MeshTriPoint & end, 
    int maxGeodesicIters = 5, ///< the maximum number of iterations to reduce approximate path length and convert it in geodesic path
    const VertBitSet* vertRegion = nullptr, Vector<float, VertId> * outSurfaceDistances = nullptr );

/// the algorithm to compute approximately geodesic path
enum class GeodesicPathApprox : char
{
    /// compute edge-only path by building it from start and end simultaneously
    DijkstraBiDir,
    /// compute edge-only path using A*-search algorithm
    DijkstraAStar,
    /// use Fast Marching algorithm
    FastMarching
};

/// returns intermediate points of the geodesic path from start to end, where it crosses mesh edges;
/// It is the same as calling computeGeodesicPathApprox() then reducePath()
MRMESH_API tl::expected<SurfacePath, PathError> computeGeodesicPath( const Mesh & mesh,
    const MeshTriPoint & start, const MeshTriPoint & end, GeodesicPathApprox atype,
    int maxGeodesicIters = 100 ); ///< the maximum number of iterations to reduce approximate path length and convert it in geodesic path

/// computes by given method and returns intermediate points of approximately geodesic path from start to end,
/// every next point is located in the same triangle with the previous point
MRMESH_API tl::expected<SurfacePath, PathError> computeGeodesicPathApprox( const Mesh & mesh,
    const MeshTriPoint & start, const MeshTriPoint & end, GeodesicPathApprox atype );

/// computes by Fast Marching method and returns intermediate points of approximately geodesic path from start to end, where it crosses mesh edges;
/// the path can be limited to given region: in face-format inside mp, or in vert-format in vertRegion argument
MRMESH_API tl::expected<SurfacePath, PathError> computeFastMarchingPath( const MeshPart & mp, 
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
