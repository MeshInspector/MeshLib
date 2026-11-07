#pragma once

#include "MRMeshFwd.h"
#include <cfloat>

namespace MR
{

/// \defgroup SurfacePathGroup Surface Paths

/// \defgroup SurfaceDistanceGroup Surface Distance
/// The functions in this group implement Fast marching method for approximately solving Eikonal equation on mesh.
/// \ingroup SurfacePathGroup
/// \{

/// computes path distances in mesh vertices from given start vertices, stopping when maxDist is reached;
/// start vertices connected by mesh edges make lines or regions (distances from the edges between them),
/// each isolated start vertex is a point;
/// considered paths can go either along edges or straightly within triangles
MRMESH_API VertScalars computeSurfaceDistances( const Mesh& mesh, const VertBitSet& startVertices, float maxDist = FLT_MAX, 
                                                          const VertBitSet* region = nullptr, int maxVertUpdates = 3 );

/// computes path distances in mesh vertices from given start vertices, stopping when all targetVertices or maxDist is reached;
/// start vertices connected by mesh edges make lines or regions (distances from the edges between them),
/// each isolated start vertex is a point;
/// considered paths can go either along edges or straightly within triangles
MRMESH_API VertScalars computeSurfaceDistances( const Mesh& mesh, const VertBitSet& startVertices, const VertBitSet& targetVertices,
    float maxDist = FLT_MAX, const VertBitSet* region = nullptr, int maxVertUpdates = 3 );

/// computes path distances in mesh vertices from given start vertices with values in them, stopping when maxDist is reached;
/// start vertices connected by mesh edges make lines or regions (distances from the edges between them),
/// each isolated start vertex is a point;
/// considered paths can go either along edges or straightly within triangles
MRMESH_API VertScalars computeSurfaceDistances( const Mesh& mesh, const HashMap<VertId, float>& startVertices, float maxDist = FLT_MAX, 
                                                          const VertBitSet* region = nullptr, int maxVertUpdates = 3 );

/// computes path distance in mesh vertices from given start point, stopping when all vertices in the face where end is located are reached;
/// distances are from the point
/// \details considered paths can go either along edges or straightly within triangles
/// \param endReached if pointer provided it will receive whether a path from start to end exists
MRMESH_API VertScalars computeSurfaceDistances( const Mesh& mesh, const MeshTriPoint & start, const MeshTriPoint & end, 
    const VertBitSet* region = nullptr, bool * endReached = nullptr, int maxVertUpdates = 3 );

/// computes path distances in mesh vertices from given start point, stopping when maxDist is reached;
/// distances are from the point;
/// considered paths can go either along edges or straightly within triangles
MRMESH_API VertScalars computeSurfaceDistances( const Mesh& mesh, const MeshTriPoint & start, float maxDist = FLT_MAX,
                                                         const VertBitSet* region = nullptr, int maxVertUpdates = 3 );

/// computes path distances in mesh vertices from given start points, stopping when maxDist is reached;
/// distances are from the nearest point, each start is a separate point even if they are close;
/// considered paths can go either along edges or straightly within triangles
MRMESH_API VertScalars computeSurfaceDistances( const Mesh& mesh, const std::vector<MeshTriPoint>& starts, float maxDist = FLT_MAX,
                                                         const VertBitSet* region = nullptr, int maxVertUpdates = 3 );

/// computes path distances in mesh vertices from given start points,
/// stopping when all vertices of all faces containing \param ends are reached;
/// distances are from the nearest point, each start is a separate point even if they are close;
/// considered paths can go either along edges or straightly within triangles
/// \param endReached if pointer provided it will receive whether a path from start to end exists
MRMESH_API VertScalars computeSurfaceDistances( const Mesh& mesh, const std::vector<MeshTriPoint>& starts, const std::vector<MeshTriPoint>& ends,
    const VertBitSet* region = nullptr, bool * endReached = nullptr, int maxVertUpdates = 3 );

/// \}

} // namespace MR
