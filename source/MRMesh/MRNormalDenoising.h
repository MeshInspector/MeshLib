#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// Smooth face normals, given
/// \param mesh contains topology information and edge lengths for weights
/// \param normals input noisy normals and output smooth normals
/// \param v edge indicator: 1 - smooth edge, 0 - crease edge
/// \param gamma the more the value, the larger smoothing is (approximately in the range [1,30])
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
MRMESH_API void denoiseNormals( const Mesh & mesh, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma );

} //namespace MR
