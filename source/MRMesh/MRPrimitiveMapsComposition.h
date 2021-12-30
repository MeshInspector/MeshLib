#pragma once
#include "MRMeshFwd.h"

namespace MR
{

// updates a2b map to a2c map using b2c map
MRMESH_API void vertMapsComposition( VertMap& a2b, const VertMap& b2c );
// returns map a2c from a2b and b2c maps
[[nodiscard]] MRMESH_API VertMap vertMapsComposition( const VertMap& a2b, const VertMap& b2c );

// updates a2b map to a2c map using b2c map
MRMESH_API void edgeMapsComposition( EdgeMap& a2b, const EdgeMap& b2c );
// returns map a2c from a2b and b2c maps
[[nodiscard]] MRMESH_API EdgeMap edgeMapsComposition( const EdgeMap& a2b, const EdgeMap& b2c );

// updates a2b map to a2c map using b2c map
MRMESH_API void faceMapsComposition( FaceMap& a2b, const FaceMap& b2c );
// returns map a2c from a2b and b2c maps
[[nodiscard]] MRMESH_API FaceMap faceMapsComposition( const FaceMap& a2b, const FaceMap& b2c );

}
