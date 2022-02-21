#pragma once
#include "MRMeshFwd.h"

namespace MR
{

/**
 * @brief fill holes with border in same plane (i.e. after cut by plane)
 * @param mesh - mesh with holes
 * @param holesEdges - edges set represents holes borders that should be filled
 * should be not empty
 * edges should have invalid left face (FaceId == -1)
 * @return true if holes filled, otherwise - false
 */
MRMESH_API bool fillContours2D( Mesh& mesh, const EdgePath& holesEdges );

}
