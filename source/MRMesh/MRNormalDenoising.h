#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// Smooth face normals, given
/// \param mesh topology and edge lengths
/// \param normals input noisy normals and output smooth normals
/// \param v edge indicator: 1 - smooth edge, 0 - crease edge
/// \param gamma amount of smoothing (e.g. in [1,30])
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
MRMESH_API void denoiseNormals( const Mesh & mesh, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma );

} //namespace MR
