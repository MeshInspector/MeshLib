#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

/// Smooth face normals, given
/// \param mesh contains topology information and coordinates for equation weights
/// \param normals input noisy normals and output smooth normals
/// \param v edge indicator function (1 - smooth edge, 0 - crease edge)
/// \param gamma the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
/// \param region if given, then only the normals of these faces are changed, and the normals of other faces act as fixed boundary conditions
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework", equation (19)
MRMESH_API void denoiseNormals( const Mesh & mesh, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma, const FaceBitSet * region = nullptr );
MRMESH_API void denoiseNormals( const MeshTopology & topology, const VertCoords & points, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma, const FaceBitSet * region = nullptr );

/// Compute edge indicator function (1 - smooth edge, 0 - crease edge) by solving large system of linear equations
/// \param mp contains topology information and coordinates for equation weights;
///           if its region is given, then only the indicator of the edges of region faces is updated, and the indicator of all other edges is fixed
/// \param normals per-face normals
/// \param beta 0.001 - sharp edges, 0.01 - moderate edges, 0.1 - smooth edges
/// \param gamma the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework", equation (20)
MRMESH_API void updateIndicator( const MeshPart & mp, Vector<float, UndirectedEdgeId> & v, const FaceNormals & normals, float beta, float gamma );

/// Compute edge indicator function (1 - smooth edge, 0 - crease edge) by approximation without solving the system of linear equations
/// \param normals per-face normals
/// \param beta 0.001 - sharp edges, 0.01 - moderate edges, 0.1 - smooth edges
/// \param gamma the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
/// \param region if given, then only the indicator of the edges of these faces is updated
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework", equation (20)
MRMESH_API void updateIndicatorFast( const MeshTopology & topology, Vector<float, UndirectedEdgeId> & v, const FaceNormals & normals, float beta, float gamma,
    const FaceBitSet * region = nullptr );

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

    /// if given, then only the normals of these faces are denoised, and only the vertices with all incident faces in the region are moved
    const FaceBitSet *region = nullptr;

    /// optionally returns creases found during smoothing, only among the edges with both incident faces in the region
    UndirectedEdgeBitSet * outCreases = nullptr;
};

/// Reduces noise in given mesh,
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
MRMESH_API void meshDenoiseViaNormals( Mesh & mesh, const DenoiseViaNormalsSettings & settings = {} );

/// the same, reporting the progress in (cb); returns false if the operation was canceled from it
[[nodiscard]] MRMESH_API bool meshDenoiseViaNormals( Mesh & mesh, const DenoiseViaNormalsSettings & settings, const ProgressCallback & cb );

struct DenoiseWithCreasesSettings
{
    /// the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
    float gamma = 5.f;

    /// how much resulting points must be attracted to initial points (e.g. to avoid general shrinkage), must be > 0
    float guideWeight = 1;

    /// the number of iterations to update vertex coordinates from found normals; the more the better quality, but longer computation
    int pointIters = 20;

    /// if given, then only the normals of these faces are denoised, and only the vertices with all incident faces in the region are moved
    const FaceBitSet *region = nullptr;
};

/// Reduces noise in given mesh, keeping the edges from (creases) sharp,
/// see the article "Mesh Denoising via a Novel Mumford-Shah Framework";
/// unlike meshDenoiseViaNormals, the creases are given by the caller and not detected automatically
MRMESH_API void meshDenoiseWithCreases( Mesh & mesh, const UndirectedEdgeBitSet & creases, const DenoiseWithCreasesSettings & settings = {} );
MRMESH_API void meshDenoiseWithCreases( const MeshTopology & topology, VertCoords & points, const UndirectedEdgeBitSet & creases, const DenoiseWithCreasesSettings & settings = {} );

/// the same, reporting the progress in (cb); returns false if the operation was canceled from it
[[nodiscard]] MRMESH_API bool meshDenoiseWithCreases( Mesh & mesh, const UndirectedEdgeBitSet & creases, const DenoiseWithCreasesSettings & settings, const ProgressCallback & cb );
[[nodiscard]] MRMESH_API bool meshDenoiseWithCreases( const MeshTopology & topology, VertCoords & points, const UndirectedEdgeBitSet & creases, const DenoiseWithCreasesSettings & settings, const ProgressCallback & cb );

} //namespace MR
