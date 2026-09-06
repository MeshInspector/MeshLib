#pragma once

#include "MRMeshFwd.h"
#include "MRMatrix3.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"

namespace MR
{

struct CameraPointsTriangulationSettings
{
    /// camera intrinsic matrix: the image-plane (pixel) position of a point p in camera space is ( K * p ).xy / ( K * p ).z
    Matrix3f intrinsics;

    /// points closer than this distance in the image plane are merged into one vertex located at their average position;
    /// non-positive value disables merging
    float weldPixels = 1;

    /// triangles having an edge longer than this in 3D are removed from the result (they typically bridge holes or depth steps);
    /// non-positive value keeps all triangles
    float maxEdgeLength = 0;

    /// to report progress and cancel
    ProgressCallback cb;
};

/// Creates a mesh from points seen by one pinhole camera (e.g. the points obtained by stereo triangulation of a single frame):
/// points are projected in the image plane, Delaunay-triangulated there (see terrainTriangulation), and the triangulation is lifted
/// back on the original 3D points, so the result is a height field over the image without self-intersections
/// \param points coordinates in the camera space (the camera is at the origin and looks along +Z), all points must have positive z
/// \return mesh with triangles oriented toward the camera
[[nodiscard]] MRMESH_API Expected<Mesh> triangulateCameraPoints( const VertCoords & points, const CameraPointsTriangulationSettings & settings );

} //namespace MR
