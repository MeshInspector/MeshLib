#pragma once

#include "MRExpected.h"
#include "MRMeshTriPoint.h"
#include "MREnums.h"
#include <vector>
#include <string>

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

inline std::string toString(PathError error)
{
    switch (error)
    {
        case PathError::StartEndNotConnected:   return "No path can be found from start to end, because they are not from the same connected component";
        case PathError::InternalError:          return "Report to developers for further investigations";
        default:                                return "Unknown error. Please, report to developers for further investigations";
    }
}

/// returns intermediate points of the geodesic path from start to end, where it crosses mesh edges;
/// the path can be limited to given region: in face-format inside mp, or in vert-format in vertRegion argument.
/// It is the same as calling computeFastMarchingPath() then reducePath()
MRMESH_API Expected<SurfacePath, PathError> computeSurfacePath( const MeshPart & mp, 
    const MeshTriPoint & start, const MeshTriPoint & end, 
    int maxGeodesicIters = 5, ///< the maximum number of iterations to reduce approximate path length and convert it in geodesic path
    const VertBitSet* vertRegion = nullptr, VertScalars * outSurfaceDistances = nullptr );

/// returns intermediate points of the geodesic path from start to end, where it crosses mesh edges;
/// It is the same as calling computeGeodesicPathApprox() then reducePath()
MRMESH_API Expected<SurfacePath, PathError> computeGeodesicPath( const Mesh & mesh,
    const MeshTriPoint & start, const MeshTriPoint & end, GeodesicPathApprox atype = GeodesicPathApprox::FastMarching,
    int maxGeodesicIters = 100 ); ///< the maximum number of iterations to reduce approximate path length and convert it in geodesic path

/// computes by given method and returns intermediate points of approximately geodesic path from start to end,
/// every next point is located in the same triangle with the previous point
MRMESH_API Expected<SurfacePath, PathError> computeGeodesicPathApprox( const Mesh & mesh,
    const MeshTriPoint & start, const MeshTriPoint & end, GeodesicPathApprox atype );

/// computes by Fast Marching method and returns intermediate points of approximately geodesic path from start to end, where it crosses mesh edges;
/// the path can be limited to given region: in face-format inside mp, or in vert-format in vertRegion argument
MRMESH_API Expected<SurfacePath, PathError> computeFastMarchingPath( const MeshPart & mp, 
    const MeshTriPoint & start, const MeshTriPoint & end, const VertBitSet* vertRegion = nullptr,
    VertScalars * outSurfaceDistances = nullptr );

struct ComputeSteepestDescentPathSettings
{
    /// if valid, then the descent is stopped as soon as same triangle with (end) is reached
    MeshTriPoint end;
    /// if not nullptr, then the descent is stopped as soon as any vertex is reached, which is written in *outVertexReached
    VertId * outVertexReached = nullptr;
    /// if not nullptr, then the descent is stopped as soon as any boundary point is reached, which is written in *outBdReached
    EdgePoint * outBdReached = nullptr;
};

/// computes the path (edge points crossed by the path) staring in given point
/// and moving in each triangle in minus gradient direction of given field;
/// the path stops when it reaches a local minimum in the field or one of stop conditions in settings
[[nodiscard]] MRMESH_API SurfacePath computeSteepestDescentPath( const MeshPart & mp, const VertScalars & field,
    const MeshTriPoint & start, const ComputeSteepestDescentPathSettings & settings = {} );

/// computes the path (edge points crossed by the path) staring in given point
/// and moving in each triangle in minus gradient direction of given field,
/// and outputs the path in \param outPath if requested;
/// the path stops when it reaches a local minimum in the field or one of stop conditions in settings
MRMESH_API void computeSteepestDescentPath( const MeshPart & mp, const VertScalars & field,
    const MeshTriPoint & start, SurfacePath * outPath, const ComputeSteepestDescentPathSettings & settings = {} );

/// finds the point along minus maximal gradient on the boundary of first ring boundary around given vertex
[[nodiscard]] MRMESH_API MeshEdgePoint findSteepestDescentPoint( const MeshPart & mp, const VertScalars & field, VertId v );

/// finds the point along minus maximal gradient on the boundary of triangles around given point (the boundary of left and right edge triangles' union in case (ep) is inner edge point)
[[nodiscard]] MRMESH_API MeshEdgePoint findSteepestDescentPoint( const MeshPart & mp, const VertScalars & field, const MeshEdgePoint & ep );

/// finds the point along minus maximal gradient on the boundary of triangles around given point (the boundary of the triangle itself in case (tp) is inner triangle point)
[[nodiscard]] MRMESH_API MeshEdgePoint findSteepestDescentPoint( const MeshPart & mp, const VertScalars & field, const MeshTriPoint & tp );

enum class ExtremeEdgeType
{
    Ridge, // where the field not-increases both in left and right triangles
    Gorge  // where the field not-decreases both in left and right triangles
};

/// computes all edges in the mesh, where the field not-increases both in left and right triangles
[[nodiscard]] MRMESH_API UndirectedEdgeBitSet findExtremeEdges( const Mesh & mesh, const VertScalars & field, ExtremeEdgeType type );

/// for each vertex from (starts) finds the closest vertex from (ends) in geodesic sense
/// \param vertRegion consider paths going in this region only
MRMESH_API HashMap<VertId, VertId> computeClosestSurfacePathTargets( const Mesh & mesh,
    const VertBitSet & starts, const VertBitSet & ends, const VertBitSet * vertRegion = nullptr,
    VertScalars * outSurfaceDistances = nullptr );

/// returns a set of mesh lines passing via most of given vertices in auto-selected order;
/// the lines try to avoid sharp turns in the vertices
MRMESH_API SurfacePaths getSurfacePathsViaVertices( const Mesh & mesh, const VertBitSet & vs );

/// computes the length of surface path
MRMESH_API float surfacePathLength( const Mesh& mesh, const SurfacePath& surfacePath );

/// converts lines on mesh in 3D contours by computing coordinate of each point
MRMESH_API Contour3f surfacePathToContour3f( const Mesh & mesh, const SurfacePath & line );
MRMESH_API Contours3f surfacePathsToContours3f( const Mesh & mesh, const SurfacePaths & lines );

/// \}

} // namespace MR
