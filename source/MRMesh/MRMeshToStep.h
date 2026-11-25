#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"
#include <filesystem>
#include <iosfwd>

namespace MR
{

struct MeshToStepSettings
{
    /// optional edges that must separate different faces of the result, e.g. the boundaries of mesh segments
    const UndirectedEdgeBitSet* faceBoundaries = nullptr;

    /// maximum distance from a mesh vertex to the analytic surface of its face;
    /// if not positive then 2.5e-6 of the diagonal of mesh bounding box is taken, or more
    /// if the coordinates are less precise, e.g. the mesh was loaded from a text file with few decimal digits
    float tolerance = 0;

    /// name of the product written in the file
    std::string name = "mesh";

    ProgressCallback progress;
};

/// numbers of faces of every surface type found by \ref saveMeshAsStep
struct MeshToStepStats
{
    int planes = 0;
    int cylinders = 0;
    int cones = 0;
    int spheres = 0;
    int tori = 0;
    int bsplines = 0;
    /// maximum distance from a mesh vertex to a B-spline surface of its face
    float maxBSplineError = 0;
};

/// subdivides the mesh on regions lying on planes, cylinders, cones, spheres and tori,
/// approximates the remaining smooth regions by B-spline surfaces,
/// and writes all of them as trimmed faces of one shell in STEP format (AP214);
/// the mesh is expected to be a tessellation of a CAD model: its vertices must be located on the surfaces precisely
MRMESH_API Expected<MeshToStepStats> saveMeshAsStep( const Mesh& mesh, std::ostream& out, const MeshToStepSettings& settings = {} );
MRMESH_API Expected<MeshToStepStats> saveMeshAsStep( const Mesh& mesh, const std::filesystem::path& file, const MeshToStepSettings& settings = {} );

} //namespace MR
