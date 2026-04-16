#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include <cfloat>
#include <optional>

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

    /// increasing this value will lead mesh vertices to shift in the directions closer to their raw pseudo-normals, but
    /// it will increase the probability of self-intersections as well;
    /// decreasing (to a positive value) will on the contrary make the field of shift directions smoother
    float normalsTrustFactor = 1;
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

    /// if true, limits the movement of each vertex to reduce self-intersections in the mesh
    bool reduceSelfIntersections = false;

    /// only if (reduceSelfIntersections = true), avoids moving a vertex closer than this distance to another triangle
    float minThickness = 0;

    /// to report progress and cancel processing
    ProgressCallback progress;
};

/// For 3D printers: shifts every vertex with normal having negative projection on Z-axis, along Z-axis;
/// mesh's topology is preserved unchanged
/// @return false if cancelled.
MRMESH_API bool zCompensate( Mesh& mesh, const ZCompensateParams& params );

/// finds the shift along z-axis for each vertex without modifying the mesh
[[nodiscard]] MRMESH_API std::optional<VertScalars> findZcompensationShifts( const Mesh& mesh, const ZCompensateParams& params );

/// finds vertices positions of the mesh after z-compensation without modifying the mesh
[[nodiscard]] MRMESH_API std::optional<VertCoords> findZcompensatedPositions( const Mesh& mesh, const ZCompensateParams& params );

} //namespace MR
