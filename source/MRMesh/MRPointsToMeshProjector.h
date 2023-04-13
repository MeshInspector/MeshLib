#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include <string>

namespace MR
{
/// Abstract class, computes the closest point on mesh to each of given points. Pure virtual finctions must be implemented
class IPointsToMeshProjector
{
protected:
public:
    virtual ~IPointsToMeshProjector() = default;    
    /// update all data related to the referencing mesh
    virtual void updateMeshData( std::shared_ptr<const Mesh> mesh ) = 0;
    /// Computes the closest point on mesh to each of given points
    virtual void findProjections( std::vector<MeshProjectionResult>& result, const std::vector<Vector3f>& points, 
                                  const AffineXf3f& worldXf, const AffineXf3f& worldRefXf, 
                                  float upDistLimitSq, float loDistLimitSq ) = 0;
};
/// Computes the closest point on mesh to each of given points on CPU
class PointsToMeshProjector : public IPointsToMeshProjector
{
    std::shared_ptr<const Mesh> mesh_;
    AffineXf3f xf_;
    AffineXf3f* refXfPtr_ = nullptr;
public:
    /// update all data related to the referencing mesh
    MRMESH_API virtual void updateMeshData( std::shared_ptr<const Mesh> mesh ) override;
    /// Computes the closest point on mesh to each of given points
    MRMESH_API virtual void findProjections( std::vector<MeshProjectionResult>& result, const std::vector<Vector3f>& points, 
                                             const AffineXf3f& objXf, const AffineXf3f& refObjXf, 
                                             float upDistLimitSq, float loDistLimitSq ) override;
};

}