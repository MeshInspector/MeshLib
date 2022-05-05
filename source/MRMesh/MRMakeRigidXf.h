#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// given a mesh part and its arbitrary transformation, computes and returns
/// the rigid transformation that best approximates meshXf
/// \ingroup MathGroup
[[nodiscard]] MRMESH_API AffineXf3d makeRigidXf( const MeshPart & mp, const AffineXf3d & meshXf );
[[nodiscard]] MRMESH_API AffineXf3f makeRigidXf( const MeshPart & mp, const AffineXf3f & meshXf );

} //namespace MR
