#pragma once
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

/// \defgroup MeshRelaxGroup Mesh Relax
/// \ingroup MeshAlgorithmGroup
/// \{

struct RelaxParams
{
    /// number of iterations
    int iterations{ 1 };
    /// region to relax
    const VertBitSet* region{ nullptr };
    /// speed of relaxing, typical values (0.0, 0.5]
    float force{ 0.5f };
};

struct MeshRelaxParams : RelaxParams
{
    /// smooth tetrahedron verts (with complete three edges ring) to base triangle (based on its edges destinations)
    bool hardSmoothTetrahedrons{ false };
};

/// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relax( Mesh& mesh, const MeshRelaxParams& params = {}, ProgressCallback cb = {} );

/// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified ) \n
/// do not really keeps volume but tries hard
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relaxKeepVolume( Mesh& mesh, const MeshRelaxParams& params = {}, ProgressCallback cb = {} );

enum class RelaxApproxType 
{
    Planar,
    Quadric
};

struct MeshApproxRelaxParams : MeshRelaxParams
{
    /// radius to find neighbors by surface
    /// 0.0f - default = 1e-3 * sqrt(surface area)
    float surfaceDilateRadius{ 0.0f };
    RelaxApproxType type{ RelaxApproxType::Planar };
};

/// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
/// approx neighborhoods
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relaxApprox( Mesh& mesh, const MeshApproxRelaxParams& params = {}, ProgressCallback cb = {} );

/// applies at most given number of relaxation iterations the spikes detected by given threshold
MRMESH_API void removeSpikes( Mesh & mesh, int maxIterations, float minSumAngle, const VertBitSet * region = nullptr );

/// \}

}
