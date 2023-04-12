#pragma once
#include "exports.h"
#include "MRMesh/MRMeshProjector.h"

namespace MR { namespace Cuda {

/// Computes distance of 2d contours according to ContourToDistanceMapParams
struct MeshData;
class MeshProjector : public IMeshProjector
{
    std::shared_ptr<MeshData> meshData_;

public:
    MRCUDA_API MeshProjector();
    MRCUDA_API virtual void updateMeshData( std::shared_ptr<const Mesh> mesh, std::string& log ) override;
    MRCUDA_API virtual std::vector<MR::MeshProjectionResult> findProjections( const std::vector<Vector3f>& points, const AffineXf3f* xf, const AffineXf3f* refXfPtr, float upDistLimitSq, float loDistLimitSq, std::vector<std::string>& log ) override;
};


}}