#pragma once

#include "MRMeshFwd.h"
#include <string>
namespace MR
{

class IMeshProjector
{
public:

    virtual ~IMeshProjector() = default;
    virtual void updateMeshData( std::shared_ptr<const Mesh> mesh ) = 0;
    virtual std::vector<MeshProjectionResult> findProjections( const std::vector<Vector3f>& points, const AffineXf3f* xf, const AffineXf3f* refXfPtr, float upDistLimitSq, float loDistLimitSq ) = 0;
};

class MeshProjector : public IMeshProjector
{
    std::shared_ptr<const Mesh> mesh_;
public:

    MRMESH_API virtual void updateMeshData( std::shared_ptr<const Mesh> mesh ) override;
    MRMESH_API virtual std::vector<MeshProjectionResult> findProjections( const std::vector<Vector3f>& points, const AffineXf3f* xf, const AffineXf3f* refXfPtr, float upDistLimitSq, float loDistLimitSq ) override;
};

}