#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"
#include <filesystem>
#include <iosfwd>
#include <string>

namespace MR
{

struct MeshToStepSettings
{
    /// the surface of a face is the simplest of plane, cylinder, cone, sphere, torus (in this order),
    /// with root-mean-square distance from the vertices of the face at most 1.5 times greater than the one of any more complex type;
    /// if that distance exceeds this tolerance, then a B-spline surface is fitted instead, and taken if it is 1.5 times closer;
    /// if the tolerance is not positive then it is 2.5 times the median of these distances over all faces (the noise level of the mesh),
    /// or 1e-6 of the diagonal of the mesh bounding box if the mesh is precise
    float tolerance = 0;

    /// name of the product written in the file
    std::string name = "mesh";

    ProgressCallback progress;
};

/// numbers of faces of every surface type written by \ref saveMeshAsStep
struct MeshToStepStats
{
    int planes = 0;
    int cylinders = 0;
    int cones = 0;
    int spheres = 0;
    int tori = 0;
    int bsplines = 0;

    /// maximum distance from a mesh vertex to the surface of its face
    float maxDeviation = 0;
};

/// writes the mesh in STEP format (AP214) as a boundary representation with one face per mesh segment,
/// where the segments are the connected parts of the mesh separated by given edges, e.g. selected by the user;
/// the surface of every face is a plane, cylinder, cone, sphere or torus fitted to the vertices of its segment, or a B-spline surface if none fits;
/// the edges of faces are the curves along the boundaries of the segments, where the surfaces of neighbour segments meet
MRMESH_API Expected<MeshToStepStats> saveMeshAsStep( const Mesh& mesh, const UndirectedEdgeBitSet& faceBoundaries, std::ostream& out,
    const MeshToStepSettings& settings = {} );
MRMESH_API Expected<MeshToStepStats> saveMeshAsStep( const Mesh& mesh, const UndirectedEdgeBitSet& faceBoundaries, const std::filesystem::path& file,
    const MeshToStepSettings& settings = {} );

} //namespace MR
