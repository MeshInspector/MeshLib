#pragma once
#include "MRMeshFwd.h"
#include "MRExpected.h"
#include <functional>

namespace MR
{

/// <summary>
/// Returns contours in \p mesh space that are offset from \p surfaceLine on \p offset amount in all directions
/// </summary>
/// <param name="mesh">mesh to perform offset on</param>
/// <param name="surfaceLine">surface line to perofrm offset from</param>
/// <param name="offset">amount of offset, note that absolute value is used</param>
/// <returns>resulting offset contours or error if something goes wrong</returns>
[[nodiscard]]
MRMESH_API Expected<Contours3f> offsetSurfaceLine( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine, float offset );

/// <summary>
/// Returns contours in \p mesh space that are offset from \p surfaceLine on \p offsetAtPoint amount in all directions
/// </summary>
/// <param name="mesh">mesh to perform offset on</param>
/// <param name="surfaceLine">surface line to perofrm offset from</param>
/// <param name="offsetAtPoint">function that can return different amount of offset in different point (argument is index of point in \p surfaceLine), note that absolute value is used</param>
/// <returns>resulting offset contours or error if something goes wrong</returns>
[[nodiscard]]
MRMESH_API Expected<Contours3f> offsetSurfaceLine( const Mesh& mesh, const std::vector<MeshTriPoint>& surfaceLine, const std::function<float( int )>& offsetAtPoint );

}