#pragma once
#include "exports.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRMesh.h"
namespace MR
{
namespace Cuda
{
struct FastWindingNumberData;
class FastWindingNumber : public IFastWindingNumber
{
    std::shared_ptr<FastWindingNumberData> data_;

public:
    MRCUDA_API FastWindingNumber( const Mesh& mesh );
    MRCUDA_API void calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace = {} ) override;
    MRCUDA_API void calcSelfIntersections( FaceBitSet& res, float beta ) override;
    MRCUDA_API void calcFromGrid( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta ) override;
};

}
}
