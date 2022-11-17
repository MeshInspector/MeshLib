#pragma once
#include "MRMeshFwd.h"
#include "MRLaplacian.h"

namespace MR
{

/// Puts given vertices in such positions to make smooth surface both inside verts-region and on its boundary;
/// \param verts must not include all vertices of a mesh connected component
MRMESH_API void positionVertsSmoothly( Mesh& mesh, const VertBitSet& verts, Laplacian::EdgeWeights egdeWeightsType = Laplacian::EdgeWeights::Cotan );

}
