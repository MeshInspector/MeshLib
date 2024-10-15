#pragma once
#include "exports.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRMesh.h"

namespace MR
{

namespace Cuda
{

struct FastWindingNumberData;

/// the class for fast approximate computation of winding number for a mesh (using its AABB tree)
/// \ingroup AABBTreeGroup
class FastWindingNumber : public IFastWindingNumber
{
    const Mesh & mesh_;
    std::shared_ptr<FastWindingNumberData> data_;

public:
    /// constructs this from AABB tree of given mesh;
    MRCUDA_API FastWindingNumber( const Mesh& mesh );

    // see methods' descriptions in IFastWindingNumber
    MRCUDA_API void calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace = {} ) override;
    MRCUDA_API bool calcSelfIntersections( FaceBitSet& res, float beta, ProgressCallback cb ) override;
    MRCUDA_API Expected<void> calcFromGrid( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, float beta, ProgressCallback cb ) override;
    MRCUDA_API float calcWithDistances( const Vector3f& p, float windingNumberThreshold, float beta, float maxDistSq, float minDistSq );
    MRCUDA_API Expected<void> calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, float windingNumberThreshold, float beta, float maxDistSq, float minDistSq, ProgressCallback cb ) override;

private:
    bool prepareData_( ProgressCallback cb );
};

} // namespace Cuda
} // namespace MR
