#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include <cfloat>

namespace MR
{

/// Modifies \p mesh shifting each vertex along its pseudonormal by the corresponding \p offset
/// @return false if cancelled.
MRMESH_API bool offsetVerts( Mesh& mesh, const VertMetric& offset, const ProgressCallback& cb = {} );

struct ThickenParams
{
    /// the amount of offset for original mesh vertices
    float outsideOffset = 0;

    /// the amount of offset for cloned mirrored mesh vertices in the opposite direction
    float insideOffset = 0;
};

/// given a mesh \p m, representing a surface,
/// creates new closed mesh by cloning mirrored mesh, and shifting original part and cloned part in different directions according to \p params,
/// if original mesh was open then stitches corresponding boundaries of two parts
MRMESH_API Mesh makeThickMesh( const Mesh & m, const ThickenParams & params );

struct ZCompensateParams
{
    /// shift of mesh parts orthogonal to Z-axis with normal against Z-axis;
    /// for other mesh parts the shift will be less and will depend on the angle between point pseudo-normal and Z-axis
    float maxShift = 0;

    /// to report progress and cancel processing
    ProgressCallback progress;
};

/// For 3D printers: shifts every vertex with normal having negative projection on Z-axis, along Z-axis
/// @return false if cancelled.
MRMESH_API bool zCompensate( Mesh& mesh, const ZCompensateParams& params );
MRMESH_API bool zCompensate( const MeshTopology& topology, VertCoords& points, const ZCompensateParams& params );

} //namespace MR
