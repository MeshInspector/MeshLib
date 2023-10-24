#pragma once
#include "exports.h"
#include "MRMesh/MRPointsToMeshProjector.h"

namespace MR { namespace Cuda {

struct MeshProjectorData;
/// Computes the closest point on mesh to each of given points on GPU. It caches data that necessary for computing
class PointsToMeshProjector : public IPointsToMeshProjector
{
    std::shared_ptr<MeshProjectorData> meshData_;
    const Mesh* mesh_{ nullptr };

public:
    MRCUDA_API PointsToMeshProjector();
    /// update all data related to the referencing mesh
    MRCUDA_API virtual void updateMeshData( const Mesh* mesh ) override;
    /// <summary>
    /// Computes the closest point on mesh to each of given points    
    /// </summary>
    /// <param name="res">vector pf projections</param>
    /// <param name="points">vector of points to project</param>
    /// <param name="objXf">transform applied to points</param>
    /// <param name="refObjXf">transform applied to referencing mesh</param>
    /// <param name="upDistLimitSq">maximal squared distance from point to mesh</param>
    /// <param name="loDistLimitSq">minimal squared distance from point to mesh</param>
    MRCUDA_API virtual void findProjections( std::vector<MR::MeshProjectionResult>& res, const std::vector<Vector3f>& points, const AffineXf3f* objXf, const AffineXf3f* refObjXf, float upDistLimitSq, float loDistLimitSq ) override;

    /// Returns amount of additional memory needed to compute projections
    MRCUDA_API virtual size_t projectionsHeapBytes( size_t numProjections ) const override;
};


}}