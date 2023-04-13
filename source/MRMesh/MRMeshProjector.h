#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include <string>

namespace MR
{
/// Abstract class, computes the closest point on mesh to each of given points. Pure virtual finctions must be implemented
class IMeshProjector
{
public:

    virtual ~IMeshProjector() = default;
    /// update transforms applied to the points and to the referencing mesh
    virtual void updateTransforms( const AffineXf3f& worldXf, const AffineXf3f& worldRefXf ) = 0;
    /// update all data related to the referencing mesh
    virtual void updateMeshData( std::shared_ptr<const Mesh> mesh ) = 0;
    /// Computes the closest point on mesh to each of given points
    virtual std::vector<MeshProjectionResult> findProjections( const std::vector<Vector3f>& points, float upDistLimitSq, float loDistLimitSq ) = 0;
};
/// Computes the closest point on mesh to each of given points on CPU
class MeshProjector : public IMeshProjector
{
    std::shared_ptr<const Mesh> mesh_;
    AffineXf3f xf_;
    AffineXf3f* refXfPtr_ = nullptr;
public:
    /// update transforms applied to the points and to the referencing mesh
    MRMESH_API virtual void updateTransforms( const AffineXf3f& worldXf, const AffineXf3f& worldRefXf );
    /// update all data related to the referencing mesh
    MRMESH_API virtual void updateMeshData( std::shared_ptr<const Mesh> mesh ) override;
    /// Computes the closest point on mesh to each of given points
    MRMESH_API virtual std::vector<MeshProjectionResult> findProjections( const std::vector<Vector3f>& points, float upDistLimitSq, float loDistLimitSq ) override;
};

}