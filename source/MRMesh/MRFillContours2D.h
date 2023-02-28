#pragma once

#include "MRExpected.h"

namespace MR
{

/**
 * @brief fill holes with border in same plane (i.e. after cut by plane)
 * @param mesh - mesh with holes
 * @param holeRepresentativeEdges - each edge here represents a hole borders that should be filled
 * should be not empty
 * edges should have invalid left face (FaceId == -1)
 * @return tl::expected with has_value()=true if holes filled, otherwise - string error
 */
MRMESH_API VoidOrErrStr fillContours2D( Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges );

}
