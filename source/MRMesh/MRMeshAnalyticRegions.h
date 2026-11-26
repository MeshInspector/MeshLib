#pragma once

#include "MRMeshFwd.h"
#include "MRCone3.h"
#include "MRCylinder3.h"
#include "MRExpected.h"
#include "MRId.h"
#include "MRPlane3.h"
#include "MRProgressCallback.h"
#include "MRSphere.h"
#include "MRTorus3.h"
#include "MRVector.h"
#include <variant>

namespace MR
{

/// a surface that findAnalyticRegions can find in a mesh:
/// - the plane has unit normal;
/// - the cylinder has unit direction, and its center and length span the vertices of the region along the axis;
/// - the cone has unit direction from its apex to the region, and its height spans the vertices of the region;
/// - the torus has unit axis
using AnalyticSurface = std::variant<Plane3d, Cylinder3d, Cone3d, Sphere3d, Torus3d>;

struct FindAnalyticRegionsSettings
{
    /// optional edges that must separate regions: no region contains triangles from both sides of such edge
    const UndirectedEdgeBitSet* faceBoundaries = nullptr;

    /// the maximum distance from the vertices of a region to its surface;
    /// if not positive, then 2.5e-6 of the diagonal of the mesh bounding box is taken, or more if the coordinates of the mesh are not precise:
    /// rounded to float or to few decimal digits (as after loading from a text file)
    float tolerance = 0;
};

/// the subdivision of a mesh on regions, each lying on its analytic surface
struct MeshAnalyticRegions
{
    /// the surface of each region
    Vector<AnalyticSurface, RegionId> surfaces;

    /// the region of each valid face of the mesh; the triangles that fit no curved surface with their neighbours
    /// are in regions of one or several triangles with their plane
    Face2RegionMap face2region;

    /// the tolerance actually used, see FindAnalyticRegionsSettings::tolerance
    float tolerance = 0;
};

/// subdivides the mesh on regions lying on planes, cylinders, cones, spheres and tori within the tolerance;
/// it is meant for meshes that are tessellations of CAD models, with the vertices precisely on the surfaces of the model:
/// the regions are grown from seed triangles over the surfaces fitted to several triangles around them, in two passes,
/// where the second pass reuses the surfaces and the axes found in the first one;
/// a prism and a coarsely tessellated cylinder are told apart by the angles between adjacent facets (35 degrees);
/// the regions are returned as they were grown: several regions can be on the same surface, a surface can be found as pieces
/// on other surfaces fitting them exactly (e.g. the rings of quadrangles of a torus on cones), and freeform parts of the mesh
/// give many small regions; uniting and fixing them is not done here;
/// \return an error if the mesh has no triangles or zero size, or the operation is canceled
[[nodiscard]] MRMESH_API Expected<MeshAnalyticRegions> findAnalyticRegions( const Mesh& mesh, const FindAnalyticRegionsSettings& settings = {},
    const ProgressCallback& progress = {} );

} //namespace MR
