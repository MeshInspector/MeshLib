#pragma once
#include "exports.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRDistanceToMeshOptions.h" // only for bindings generation

namespace MR
{

namespace Cuda
{

struct FastWindingNumberData;

/// the class for fast approximate computation of winding number for a mesh (using its AABB tree)
/// \ingroup AABBTreeGroup
class MRCUDA_CLASS FastWindingNumber : public IFastWindingNumber
{
    const Mesh & mesh_;
    std::shared_ptr<FastWindingNumberData> data_;

public:
    /// constructs this from AABB tree of given mesh;
    MRCUDA_API FastWindingNumber( const Mesh& mesh );

    // see methods' descriptions in IFastWindingNumber
    MRCUDA_API Expected<void> calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace, const ProgressCallback& cb ) override;
    MRCUDA_API Expected<void> calcSelfIntersections( FaceBitSet& res, float beta, const ProgressCallback& cb ) override;
    MRCUDA_API Expected<void> calcFromGrid( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, float beta, const ProgressCallback& cb ) override;
    MRCUDA_API Expected<void> calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, const DistanceToMeshOptions& options, const ProgressCallback& cb ) override;

private:
    Expected<void> prepareData_( ProgressCallback cb );
};

} // namespace Cuda
} // namespace MR
