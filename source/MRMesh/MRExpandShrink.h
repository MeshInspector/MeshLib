#pragma once

#include "MRMeshFwd.h"

namespace MR
{

// adds to the region all faces within given number of hops (stars) from the initial region boundary
MRMESH_API void expand( const MeshTopology & topology, FaceBitSet & region, int hops = 1 );
// returns the region of all faces within given number of hops (stars) from the initial face
[[nodiscard]] MRMESH_API FaceBitSet expand( const MeshTopology & topology, FaceId f, int hops );

// adds to the region all vertices within given number of hops (stars) from the initial region boundary
MRMESH_API void expand( const MeshTopology & topology, VertBitSet & region, int hops = 1 );
// returns the region of all vertices within given number of hops (stars) from the initial vertex
[[nodiscard]] MRMESH_API VertBitSet expand( const MeshTopology & topology, VertId v, int hops );

// removes from the region all faces within given number of hops (stars) from the initial region boundary
MRMESH_API void shrink( const MeshTopology & topology, FaceBitSet & region, int hops = 1 );
// removes from the region all vertices within given number of hops (stars) from the initial region boundary
MRMESH_API void shrink( const MeshTopology & topology, VertBitSet & region, int hops = 1 );

} //namespace MR
