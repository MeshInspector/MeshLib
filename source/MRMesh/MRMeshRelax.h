#pragma once
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRRelaxParams.h"

namespace MR
{

/// \defgroup MeshRelaxGroup Mesh Relax
/// \ingroup MeshAlgorithmGroup
/// \{

struct MeshRelaxParams : RelaxParams
{
    /// move all region vertices with exactly three neighbor vertices in the center of the neighbors
    bool hardSmoothTetrahedrons{ false };

    /// weight for each vertex. By default, all the vertices have equal weights.
    const VertScalars *weights = nullptr;
};

/// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relax( Mesh& mesh, const MeshRelaxParams& params = {}, ProgressCallback cb = {} );

/// computes position of a vertex, when all neighbor triangles have almost equal areas,
/// more precisely it minimizes sum_i (area_i)^2 by adjusting the position of this vertex only
[[nodiscard]] MRMESH_API Vector3f vertexPosEqualNeiAreas( const Mesh& mesh, VertId v, bool noShrinkage );

struct MeshEqualizeTriAreasParams : MeshRelaxParams
{
    /// if true prevents the surface from shrinkage after many iterations;
    /// technically it is done by solving the same task in the plane orthogonal to normal direction
    bool noShrinkage{ false };
};

/// applies given number of iterations with movement toward vertexPosEqualNeiAreas() to the whole mesh ( or some region if it is specified )
/// \return true if the operation completed successfully, and false if it was interrupted by the progress callback.
MRMESH_API bool equalizeTriAreas( Mesh& mesh, const MeshEqualizeTriAreasParams& params = {}, ProgressCallback cb = {} );

/// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified ) \n
/// do not really keeps volume but tries hard
/// \return true if the operation completed successfully, and false if it was interrupted by the progress callback.
MRMESH_API bool relaxKeepVolume( Mesh& mesh, const MeshRelaxParams& params = {}, ProgressCallback cb = {} );

struct MeshApproxRelaxParams : MeshRelaxParams
{
    /// radius to find neighbors by surface
    /// 0.0f - default = 1e-3 * sqrt(surface area)
    float surfaceDilateRadius{ 0.0f };
    RelaxApproxType type{ RelaxApproxType::Planar };
};

/// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
/// approx neighborhoods
/// \return true if the operation completed successfully, and false if it was interrupted by the progress callback.
MRMESH_API bool relaxApprox( Mesh& mesh, const MeshApproxRelaxParams& params = {}, ProgressCallback cb = {} );

/// applies at most given number of relaxation iterations the spikes detected by given threshold
MRMESH_API void removeSpikes( Mesh & mesh, int maxIterations, float minSumAngle, const VertBitSet * region = nullptr );

/// given a region of faces on the mesh, moves boundary vertices of the region
/// to make the region contour much smoother with minor optimization of mesh topology near region boundary;
/// \param numIters >= 1 how many times to run the algorithm to achieve a better quality,
/// solution is typically oscillates back and forth so even number of iterations is recommended
MRMESH_API void smoothRegionBoundary( Mesh & mesh, const FaceBitSet & regionFaces, int numIters = 4 );

/// move all region vertices with exactly three neighbor vertices in the center of the neighbors
MRMESH_API void hardSmoothTetrahedrons( Mesh & mesh, const VertBitSet *region = nullptr );

/// \}

}
