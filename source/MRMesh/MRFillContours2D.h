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
 * @return Expected with has_value()=true if holes filled, otherwise - string error
 */
MRMESH_API Expected<void> fillContours2D( Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges );

/// computes the transformation that maps
/// O into center mass of contours' points
/// OXY into best plane containing the points
MRMESH_API AffineXf3f getXfFromOxyPlane( const Contours3f& contours );
MRMESH_API AffineXf3f getXfFromOxyPlane( const Mesh& mesh, const std::vector<EdgePath>& paths );

}
