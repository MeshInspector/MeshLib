#pragma once
#include "exports.h"
#include "MRMesh/MRPointsToMeshProjector.h"

namespace MR { namespace Cuda {

struct MeshProjectorData;
/// Computes the closest point on mesh to each of given points on GPU
class PointsToMeshProjector : public IPointsToMeshProjector
{
    std::shared_ptr<MeshProjectorData> meshData_;

public:
    MRCUDA_API PointsToMeshProjector();
    /// update transforms applied to the points and to the referencing mesh
    MRCUDA_API virtual void updateTransforms( const AffineXf3f& objXf, const AffineXf3f& refObjXf ) override;
    /// update all data related to the referencing mesh
    MRCUDA_API virtual void updateMeshData( std::shared_ptr<const Mesh> mesh ) override;
    /// Computes the closest point on mesh to each of given points
    MRCUDA_API virtual std::vector<MR::MeshProjectionResult> findProjections( const std::vector<Vector3f>& points, float upDistLimitSq, float loDistLimitSq ) override;
};


}}