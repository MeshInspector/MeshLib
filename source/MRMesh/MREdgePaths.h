#pragma once

#include "MRMeshFwd.h"
#include "MRMesh/MRProgressCallback.h"
#include <cfloat>
#include <functional>
#include <vector>

namespace MR
{

/// \defgroup SurfacePathGroup

/// \defgroup EdgePathsGroup Edge Paths
/// \ingroup SurfacePathGroup
/// \{

using EdgeMetric = std::function<float( EdgeId )>;

/// returns true if every next edge starts where previous edge ends
[[nodiscard]] MRMESH_API bool isEdgePath( const MeshTopology & topology, const std::vector<EdgeId> & edges );

/// Same as \ref isEdgePath, but start vertex coincide with finish vertex
[[nodiscard]] MRMESH_API bool isEdgeLoop( const MeshTopology & topology, const std::vector<EdgeId> & edges );

/// reverses the order of edges and flips each edge orientation, thus
/// making the opposite directed edge path
MRMESH_API void reverse( EdgePath & path );
/// reverse every path in the vector
MRMESH_API void reverse( std::vector<EdgePath> & paths );

/// metric returning 1 for every edge
[[nodiscard]] MRMESH_API EdgeMetric identityMetric();

/// returns edge's length as a metric
[[nodiscard]] MRMESH_API EdgeMetric edgeLengthMetric( const Mesh & mesh );

/// returns edge's metric that depends both on edge's length and on the angle between its left and right faces
/// \param angleSinFactor multiplier before dihedral angle sine in edge metric calculation (positive to prefer concave angles, negative - convex)
/// \param angleSinForBoundary consider this dihedral angle sine for boundary edges
[[nodiscard]] MRMESH_API EdgeMetric edgeCurvMetric( const Mesh & mesh, float angleSinFactor = 2, float angleSinForBoundary = 0 );

/// pre-computes the metric for all mesh edges to quickly return it later for any edge
[[nodiscard]] MRMESH_API EdgeMetric edgeTableMetric( const MeshTopology & topology, const EdgeMetric & metric );

/// computes summed metric of all edges in the path
[[nodiscard]] MRMESH_API double calcPathMetric( const EdgePath & path, EdgeMetric metric );
[[nodiscard]] inline double calcPathLength( const EdgePath & path, const Mesh & mesh ) { return calcPathMetric( path, edgeLengthMetric( mesh ) ); }

/// sorts given paths in ascending order of their metrics
MRMESH_API void sortPathsByMetric( std::vector<EdgePath> & paths, EdgeMetric metric );
inline void sortPathsByLength( std::vector<EdgePath> & paths, const Mesh & mesh ) { sortPathsByMetric( paths, edgeLengthMetric( mesh ) ); }

/// adds all faces incident to loop vertices and located to the left from the loop to given FaceBitSet
MRMESH_API void addLeftBand( const MeshTopology & topology, const EdgeLoop & loop, FaceBitSet & addHere );

/// finds the shortest path in euclidean metric from start to finish vertices using Dijkstra algorithm;
/// if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPath( const Mesh & mesh, VertId start, VertId finish, float maxPathLen = FLT_MAX );
/// finds the shortest path in euclidean metric from start to finish vertices using bidirectional modification of Dijkstra algorithm,
/// constructing the path simultaneously from both start and finish, which is faster for long paths;
/// if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPathBiDir( const Mesh & mesh, VertId start, VertId finish, float maxPathLen = FLT_MAX );
/// finds the shortest path in euclidean metric from start to finish vertices using A* modification of Dijkstra algorithm,
/// which is faster for near linear path;
/// if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPathAStar( const Mesh & mesh, VertId start, VertId finish, float maxPathLen = FLT_MAX );

/// builds shortest path in euclidean metric from start to finish vertices; if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPath( const Mesh& mesh, VertId start, const VertBitSet& finish, float maxPathLen = FLT_MAX );

/// builds shortest path in given metric from start to finish vertices; if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildSmallestMetricPath( const MeshTopology & topology, const EdgeMetric & metric,
    VertId start, VertId finish, float maxPathMetric = FLT_MAX );
/// same, but constructs the path from both start and finish, which is faster for long paths
[[nodiscard]] MRMESH_API EdgePath buildSmallestMetricPathBiDir( const MeshTopology & topology, const EdgeMetric & metric,
    VertId start, VertId finish, float maxPathMetric = FLT_MAX );

/// builds shortest path in given metric from start to finish vertices; if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildSmallestMetricPath( const MeshTopology& topology, const EdgeMetric& metric,
    VertId start, const VertBitSet& finish, float maxPathMetric = FLT_MAX );

/// returns all vertices from given region ordered in each connected component in breadth-first way
[[nodiscard]] MRMESH_API std::vector<VertId> getVertexOrdering( const MeshTopology & topology, VertBitSet region );

/// finds all closed loops from given edges and removes them from edges
[[nodiscard]] MRMESH_API std::vector<EdgeLoop> extractClosedLoops( const MeshTopology & topology, EdgeBitSet & edges );
[[nodiscard]] MRMESH_API std::vector<EdgeLoop> extractClosedLoops( const MeshTopology & topology, const std::vector<EdgeId> & inEdges,
    EdgeBitSet * outNotLoopEdges = nullptr );
[[nodiscard]] MRMESH_API EdgeLoop extractLongestClosedLoop( const Mesh & mesh, const std::vector<EdgeId> & inEdges );

/// expands the region (of faces or vertices) on given metric value. returns false if callback also returns false
MRMESH_API bool dilateRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, FaceBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool dilateRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, VertBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool dilateRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback = {} );

/// shrinks the region (of faces or vertices) on given metric value. returns false if callback also returns false
MRMESH_API bool erodeRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, FaceBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool erodeRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, VertBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool erodeRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback = {} );

/// expands the region (of faces or vertices) on given value (in meters). returns false if callback also returns false
MRMESH_API bool dilateRegion( const Mesh& mesh, FaceBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool dilateRegion( const Mesh& mesh, VertBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool dilateRegion( const Mesh& mesh, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback = {} );

/// shrinks the region (of faces or vertices) on given value (in meters). returns false if callback also returns false
MRMESH_API bool erodeRegion( const Mesh& mesh, FaceBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool erodeRegion( const Mesh& mesh, VertBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool erodeRegion( const Mesh& mesh, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback = {} );

/// finds all intersection points between given path and plane, adds them in outIntersections and returns their number
MRMESH_API int getPathPlaneIntersections( const Mesh & mesh, const EdgePath & path, const Plane3f & plane,
    std::vector<MeshEdgePoint> * outIntersections = nullptr );

/// finds all path edges located in given plane with given tolerance, adds them in outInPlaneEdges and returns their number
MRMESH_API int getPathEdgesInPlane( const Mesh & mesh, const EdgePath & path, const Plane3f & plane, float tolerance = 0.0f,
    std::vector<EdgeId> * outInPlaneEdges = nullptr );

/// \}

} // namespace MR
