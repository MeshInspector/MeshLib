#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// Smooth face normals, given
/// \param mesh contains topology information and coordinates for equation weights
/// \param normals input noisy normals and output smooth normals
/// \param v edge indicator function (1 - smooth edge, 0 - crease edge)
/// \param gamma the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
MRMESH_API void denoiseNormals( const Mesh & mesh, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma );

/// Update edge indicator function (1 - smooth edge, 0 - crease edge), given
/// \param mesh contains topology information and coordinates for equation weights
/// \param v noisy input and smooth output
/// \param normals per-face normals
/// \param beta 0.001 - sharp edges, 0.01 - moderate edges, 0.1 - smooth edges
/// \param gamma the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
MRMESH_API void updateIndicator( const Mesh & mesh, Vector<float, UndirectedEdgeId> & v, const FaceNormals & normals, float beta, float gamma );

} //namespace MR
