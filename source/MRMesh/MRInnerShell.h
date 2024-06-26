#pragma once

#include "MRMeshFwd.h"
#include <cfloat>

namespace MR
{

enum class Side
{
    Negative,
    Positive
};

struct FindInnerShellSettings
{
    /// specifies which side of shell is of interest: negative or positive relative to mesh normals
    Side side = Side::Negative;

    /// specifies maximum squared distance from shell parts of interest to source mesh
    float maxDistSq = FLT_MAX;

    /// if true, a slower algorithm is activated that is more robust in the presence of self-intersections on mesh
    bool useWindingNumber = false;

    /// positive side if winding number below or equal this threshold;
    /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
    float windingNumberThreshold = 0.25f;
};

/// Tests \param shellPoint from bidirectional shell constructed for an open \param mp;
/// \return whether the distance from given point to given mesh part is of same sign as settings.side,
/// if useWindingNumber = false, returns false for all points projecting on mesh boundary
[[nodiscard]] MRMESH_API bool isInnerShellVert( const MeshPart & mp, const Vector3f & shellPoint, const FindInnerShellSettings & settings = {} );

/// Finds inner-shell vertices on bidirectional \param shell constructed for an open \param mp;
/// The function will return all shell vertices that have distance to mesh of same sign as settings.side
[[nodiscard]] MRMESH_API VertBitSet findInnerShellVerts( const MeshPart & mp, const Mesh & shell, const FindInnerShellSettings & settings = {} );

/// Finds inner-shell faces on bidirectional \param shell constructed for an open \param mp;
/// The function will return all shell faces (after some subdivision) that have distance to mesh of same sign as settings.side
[[nodiscard]] MRMESH_API FaceBitSet findInnerShellFacesWithSplits( const MeshPart & mp, Mesh & shell, const FindInnerShellSettings & settings = {} );

} // namespace MR
