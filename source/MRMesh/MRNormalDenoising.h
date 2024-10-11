#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"

namespace MR
{

/// Smooth face normals, given
/// \param mesh contains topology information and coordinates for equation weights
/// \param normals input noisy normals and output smooth normals
/// \param v edge indicator function (1 - smooth edge, 0 - crease edge)
/// \param gamma the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework", equation (19)
MRMESH_API void denoiseNormals( const Mesh & mesh, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma );

/// Compute edge indicator function (1 - smooth edge, 0 - crease edge) by solving large system of linear equations
/// \param mesh contains topology information and coordinates for equation weights
/// \param normals per-face normals
/// \param beta 0.001 - sharp edges, 0.01 - moderate edges, 0.1 - smooth edges
/// \param gamma the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework", equation (20)
MRMESH_API void updateIndicator( const Mesh & mesh, Vector<float, UndirectedEdgeId> & v, const FaceNormals & normals, float beta, float gamma );

/// Compute edge indicator function (1 - smooth edge, 0 - crease edge) by approximation without solving the system of linear equations
/// \param normals per-face normals
/// \param beta 0.001 - sharp edges, 0.01 - moderate edges, 0.1 - smooth edges
/// \param gamma the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework", equation (20)
MRMESH_API void updateIndicatorFast( const MeshTopology & topology, Vector<float, UndirectedEdgeId> & v, const FaceNormals & normals, float beta, float gamma );

struct DenoiseViaNormalsSettings
{
    /// use approximated computation, which is much faster than precise solution
    bool fastIndicatorComputation = true;

    /// 0.001 - sharp edges, 0.01 - moderate edges, 0.1 - smooth edges
    float beta = 0.01f;

    /// the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
    float gamma = 5.f;

    /// the number of iterations to smooth normals and find creases; the more the better quality, but longer computation
    int normalIters = 10;

    /// the number of iterations to update vertex coordinates from found normals; the more the better quality, but longer computation
    int pointIters = 20;

    /// how much resulting points must be attracted to initial points (e.g. to avoid general shrinkage), must be > 0
    float guideWeight = 1;

    /// if true then maximal displacement of each point during denoising will be limited
    bool limitNearInitial = false;

    /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
    float maxInitialDist = 0;

    /// optionally returns creases found during smoothing
    UndirectedEdgeBitSet * outCreases = nullptr;

    /// to get the progress and optionally cancel
    ProgressCallback cb = {};
};

/// Reduces noise in given mesh,
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
MRMESH_API Expected<void> meshDenoiseViaNormals( Mesh & mesh, const DenoiseViaNormalsSettings & settings = {} );

} //namespace MR
