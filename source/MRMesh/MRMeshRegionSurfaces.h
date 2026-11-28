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

#include <memory>
#include <variant>

namespace MR
{

struct FittedBSplineSurface;

/// a smooth surface fitted to a region of a mesh:
/// - the plane has unit normal;
/// - the cylinder has unit direction, and its center and length span the vertices of the region along the axis;
/// - the cone has unit direction from its apex to the region, and its height spans the vertices of the region;
/// - the torus has unit axis;
/// - the B-spline surface is fitted to a region with one boundary loop, where analytic surfaces are far from the vertices
using RegionSurface = std::variant<Plane3d, Cylinder3d, Cone3d, Sphere3d, Torus3d, std::shared_ptr<const FittedBSplineSurface>>;

struct FittedRegionSurface
{
    RegionSurface surface;

    /// root-mean-square distance from the vertices of the region to the surface
    float rms = 0;

    /// maximum distance from the vertices of the region to the surface
    float maxDist = 0;

    friend bool operator ==( const FittedRegionSurface& a, const FittedRegionSurface& b ) = default;
};

struct FitRegionSurfacesSettings
{
    /// if the root-mean-square distance from the vertices of a region to its best analytic surface is above this value,
    /// then a B-spline surface is fitted to the region with this maximum distance from the vertices, and taken if it is 1.5 times closer;
    /// if not positive, then 2.5 times the median of these distances over all regions is taken:
    /// most regions are expected to be analytic, and then the distances are the noise of the mesh
    float bsplineThreshold = 0;

    /// if false, then only analytic surfaces are fitted
    bool allowBSplines = true;
};

struct MeshRegionSurfaces
{
    /// the surface of every region; the regions without faces get planes with zero normals
    Vector<FittedRegionSurface, RegionId> surfaces;

    /// the threshold actually used, see FitRegionSurfacesSettings::bsplineThreshold
    float bsplineThreshold = 0;
};

/// fits a surface to every region of the mesh: the simplest of plane, cylinder, cone, sphere and torus, unless a more complex one
/// is 1.5 times closer to the vertices of the region; curved surfaces fitted to a different shape are rejected (if the normals of more than 20%
/// of the area of the region differ by more than 45 degrees from the normals of the surface, or if a surface of revolution is cut
/// by the region only on one side, e.g. a cylinder fitted to a small bump); the regions far from all analytic surfaces get B-spline surfaces;
/// the regions are given by the faces of the mesh, e.g. selected by the user or found by MeshComponents::getAllComponentsMap with separating edges;
/// \param face2region the region of every valid face of the mesh
/// \return an error if the mesh has no triangles or zero size, or the operation is canceled
[[nodiscard]] MRMESH_API Expected<MeshRegionSurfaces> fitRegionSurfaces( const Mesh& mesh, const Face2RegionMap& face2region,
    const FitRegionSurfacesSettings& settings = {}, const ProgressCallback& progress = {} );

} //namespace MR
