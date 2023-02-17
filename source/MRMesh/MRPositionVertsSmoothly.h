#pragma once
#include "MRMeshFwd.h"
#include "MRLaplacian.h"

namespace MR
{

/// Puts given vertices in such positions to make smooth surface both inside verts-region and on its boundary;
/// \param verts must not include all vertices of a mesh connected component
/// \param fixedSharpVertices in these vertices the surface can be not-smooth
MRMESH_API void positionVertsSmoothly( Mesh& mesh, const VertBitSet& verts,
    Laplacian::EdgeWeights egdeWeightsType = Laplacian::EdgeWeights::Cotan,
    const VertBitSet * fixedSharpVertices = nullptr );

/// Puts given vertices in such positions to make smooth surface inside verts-region, but sharp on its boundary;
/// \param verts must not include all vertices of a mesh connected component
MRMESH_API void positionVertsSmoothlySharpBd( Mesh& mesh, const VertBitSet& verts );

} //namespace MR
