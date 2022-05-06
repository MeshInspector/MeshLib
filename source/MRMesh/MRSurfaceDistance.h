#pragma once

#include "MRMeshFwd.h"
#include <cfloat>

namespace MR
{

/// \defgroup SurfaceDistanceGroup Surface Distance
/// \ingroup SurfacePathGroup
/// \{

/// computes path distances in mesh vertices from given start vertices, stopping when maxDist is reached;
/// considered paths can go either along edges or straightly within triangles
MRMESH_API Vector<float, VertId> computeSurfaceDistances( const Mesh& mesh, const VertBitSet& startVertices, float maxDist = FLT_MAX, 
                                                          const VertBitSet* region = nullptr );

/// computes path distances in mesh vertices from given start vertices, stopping when all targetVertices or maxDist is reached;
/// considered paths can go either along edges or straightly within triangles
MRMESH_API Vector<float, VertId> computeSurfaceDistances( const Mesh& mesh, const VertBitSet& startVertices, const VertBitSet& targetVertices,
    float maxDist = FLT_MAX, const VertBitSet* region = nullptr );

/// computes path distances in mesh vertices from given start vertices with values in them, stopping when maxDist is reached;
/// considered paths can go either along edges or straightly within triangles
MRMESH_API Vector<float, VertId> computeSurfaceDistances( const Mesh& mesh, const HashMap<VertId, float>& startVertices, float maxDist = FLT_MAX, 
                                                          const VertBitSet* region = nullptr );

/// computes path distance in mesh vertices from given start point, stopping when all vertices in the face where end is located are reached;
/// \details considered paths can go either along edges or straightly within triangles
/// \param endReached if pointer provided it will receive where a path from start to end exists
MRMESH_API Vector<float,VertId> computeSurfaceDistances( const Mesh& mesh, const MeshTriPoint & start, const MeshTriPoint & end, 
    const VertBitSet* region = nullptr, bool * endReached = nullptr );

/// \}

} // namespace MR
