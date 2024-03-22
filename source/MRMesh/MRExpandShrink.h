#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// adds to the region all faces within given number of hops (stars) from the initial region boundary
MRMESH_API void expand( const MeshTopology & topology, FaceBitSet & region, int hops = 1 );
/// returns the region of all faces within given number of hops (stars) from the initial face
[[nodiscard]] MRMESH_API FaceBitSet expand( const MeshTopology & topology, FaceId f, int hops );

// adds to the region all vertices within given number of hops (stars) from the initial region boundary
MRMESH_API void expand( const MeshTopology & topology, VertBitSet & region, int hops = 1 );
/// returns the region of all vertices within given number of hops (stars) from the initial vertex
[[nodiscard]] MRMESH_API VertBitSet expand( const MeshTopology & topology, VertId v, int hops );

/// removes from the region all faces within given number of hops (stars) from the initial region boundary
MRMESH_API void shrink( const MeshTopology & topology, FaceBitSet & region, int hops = 1 );
/// removes from the region all vertices within given number of hops (stars) from the initial region boundary
MRMESH_API void shrink( const MeshTopology & topology, VertBitSet & region, int hops = 1 );

/// returns given region with all faces sharing an edge with a region face;
/// \param stopEdges - neighborhood via this edges will be ignored
[[nodiscard]] MRMESH_API FaceBitSet expandFaces( const MeshTopology & topology, const FaceBitSet & region, const UndirectedEdgeBitSet * stopEdges = nullptr );

/// returns given region without all faces sharing an edge with not-region face;
/// \param stopEdges - neighborhood via this edges will be ignored
[[nodiscard]] MRMESH_API FaceBitSet shrinkFaces( const MeshTopology & topology, const FaceBitSet & region, const UndirectedEdgeBitSet * stopEdges = nullptr );

/// returns faces from given region that have at least one neighbor face with shared edge not from the region
[[nodiscard]] MRMESH_API FaceBitSet getBoundaryFaces( const MeshTopology & topology, const FaceBitSet & region );

} //namespace MR
