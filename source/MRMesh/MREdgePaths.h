#pragma once

#include "MRMeshFwd.h"
#include <cfloat>
#include <functional>
#include <vector>

namespace MR
{

using EdgeMetric = std::function<float( EdgeId )>;

// returns true if every next edge starts where previous edge ends
[[nodiscard]] MRMESH_API bool isEdgePath( const MeshTopology & topology, const std::vector<EdgeId> & edges );

// = isEdgePath and start vertex coincide with finish vertex
[[nodiscard]] MRMESH_API bool isEdgeLoop( const MeshTopology & topology, const std::vector<EdgeId> & edges );

// reverses the order of edges and flips each edge orientation, thus
// making the opposite directed edge path
MRMESH_API void reverse( EdgePath & path );
// reverse every path in the vector
MRMESH_API void reverse( std::vector<EdgePath> & paths );

// metric returning 1 for every edge
[[nodiscard]] MRMESH_API EdgeMetric identityMetric();

// returns edge's length as a metric
[[nodiscard]] MRMESH_API EdgeMetric edgeLengthMetric( const Mesh & mesh );

// returns edge's metric that depends both on edge's length and on the angle between its left and right faces
[[nodiscard]] MRMESH_API EdgeMetric edgeCurvMetric( const Mesh & mesh, 
    float angleSinFactor = 2,                  // multiplier before dihedral angle sine in edge metric calculation (positive to prefer concave angles, negative - convex)
    float angleSinForBoundary = 0 );           // consider this dihedral angle sine for boundary edges

// pre-computes the metric for all mesh edges to quickly return it later for any edge
[[nodiscard]] MRMESH_API EdgeMetric edgeTableMetric( const MeshTopology & topology, const EdgeMetric & metric );

// computes summed metric of all edges in the path
[[nodiscard]] MRMESH_API double calcPathMetric( const EdgePath & path, EdgeMetric metric );
[[nodiscard]] inline double calcPathLength( const EdgePath & path, const Mesh & mesh ) { return calcPathMetric( path, edgeLengthMetric( mesh ) ); }

// sorts given paths in ascending order of their metrics
MRMESH_API void sortPathsByMetric( std::vector<EdgePath> & paths, EdgeMetric metric );
inline void sortPathsByLength( std::vector<EdgePath> & paths, const Mesh & mesh ) { sortPathsByMetric( paths, edgeLengthMetric( mesh ) ); }

// adds all faces incident to loop vertices and located to the left from the loop to given FaceBitSet
MRMESH_API void addLeftBand( const MeshTopology & topology, const EdgeLoop & loop, FaceBitSet & addHere );

// builds shortest path in euclidean metric from start to finish vertices; if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPath( const Mesh & mesh, VertId start, VertId finish, float maxPathLen = FLT_MAX );

// builds shortest path in euclidean metric from start to finish vertices; if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPath( const Mesh& mesh, VertId start, const VertBitSet& finish, float maxPathLen = FLT_MAX );

// builds shortest path in given metric from start to finish vertices; if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildSmallestMetricPath( const MeshTopology & topology, const EdgeMetric & metric,
    VertId start, VertId finish, float maxPathMetric = FLT_MAX );

// builds shortest path in given metric from start to finish vertices; if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildSmallestMetricPath( const MeshTopology& topology, const EdgeMetric& metric,
    VertId start, const VertBitSet& finish, float maxPathMetric = FLT_MAX );

// returns all vertices from given region ordered in each connected component in breadth-first way
[[nodiscard]] MRMESH_API std::vector<VertId> getVertexOrdering( const MeshTopology & topology, VertBitSet region );

// finds all closed loops from given edges and removes them from edges
[[nodiscard]] MRMESH_API std::vector<EdgeLoop> extractClosedLoops( const MeshTopology & topology, EdgeBitSet & edges );
[[nodiscard]] MRMESH_API std::vector<EdgeLoop> extractClosedLoops( const MeshTopology & topology, const std::vector<EdgeId> & inEdges,
    EdgeBitSet * outNotLoopEdges = nullptr );
[[nodiscard]] MRMESH_API EdgeLoop extractLongestClosedLoop( const Mesh & mesh, const std::vector<EdgeId> & inEdges );

// expands the region (of faces or vertices) on given metric value
MRMESH_API void dilateRegionByMetric( const MeshTopology & topology, const EdgeMetric & metric, FaceBitSet & region, float dilation );
MRMESH_API void dilateRegionByMetric( const MeshTopology & topology, const EdgeMetric & metric, VertBitSet & region, float dilation );

// shrinks the region (of faces or vertices) on given metric value
MRMESH_API void erodeRegionByMetric( const MeshTopology & topology, const EdgeMetric & metric, FaceBitSet & region, float dilation );
MRMESH_API void erodeRegionByMetric( const MeshTopology & topology, const EdgeMetric & metric, VertBitSet & region, float dilation );

// expands the region (of faces or vertices) on given value (in meters)
MRMESH_API void dilateRegion( const Mesh & mesh, FaceBitSet & region, float dilation );
MRMESH_API void dilateRegion( const Mesh & mesh, VertBitSet & region, float dilation );

// shrinks the region (of faces or vertices) on given value (in meters)
MRMESH_API void erodeRegion( const Mesh & mesh, FaceBitSet & region, float dilation );
MRMESH_API void erodeRegion( const Mesh & mesh, VertBitSet & region, float dilation );

// finds all intersection points between given path and plane, adds them in outIntersections and returns their number
MRMESH_API int getPathPlaneIntersections( const Mesh & mesh, const EdgePath & path, const Plane3f & plane,
    std::vector<MeshEdgePoint> * outIntersections = nullptr );

// finds all path edges located in given plane with given tolerance, adds them in outInPlaneEdges and returns their number
MRMESH_API int getPathEdgesInPlane( const Mesh & mesh, const EdgePath & path, const Plane3f & plane, float tolerance = 0.0f,
    std::vector<EdgeId> * outInPlaneEdges = nullptr );

} //namespace MR
