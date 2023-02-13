#pragma once

#include "MRMeshFwd.h"
#include <tl/expected.hpp>
#include <string>

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
MRMESH_API tl::expected<void, std::string> fillContours2D( Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges );

}
