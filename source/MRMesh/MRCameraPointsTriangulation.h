#pragma once

#include "MRMeshFwd.h"
#include "MRMatrix3.h"
#include "MRExpected.h"
#include "MRProgressCallback.h"
#include "MRPositionVertsSmoothly.h"

namespace MR
{

struct CameraPointsTriangulationSettings
{
    /// camera intrinsic matrix: the image-plane (pixel) position of a point p in camera space is ( K * p ).xy / ( K * p ).z
    Matrix3f intrinsics;

    /// points closer than this distance in the image plane are merged into one vertex located at their average position;
    /// non-positive value disables merging
    float weldPixels = 1;

    /// if not null, receives the result of findSmallestCloseVertices on the image-plane positions:
    /// each point is mapped to the point it was merged into (or to itself); left unchanged if weldPixels <= 0
    VertMap * outSmallestMap = nullptr;

    /// if not null, receives the image-plane coordinates of all points as ( x, -y, 1 ) with y mirrored,
    /// for welded points these are the coordinates of the averaged position, for invalid cloud points zeros
    VertCoords * outProjectedPoints = nullptr;
};

/// Creates a mesh from points seen by one pinhole camera (e.g. the points obtained by stereo triangulation of a single frame):
/// points are projected in the image plane, Delaunay-triangulated there (see delaunayTriangulationXY), and the triangulation is lifted
/// back on the original 3D points, so the result is a height field over the image without self-intersections
/// \param points coordinates in the camera space (the camera is at the origin and looks along +Z), all points must have positive z
/// \return mesh with triangles oriented toward the camera; vertex ids are the same as in \p points, and the points merged
///         into others by welding become invalid vertices; the triangles bridging holes in the sampling or depth steps
///         can be removed afterwards by deleteFacesWithLongEdges
[[nodiscard]] MRMESH_API Expected<Mesh> triangulateCameraPoints( const VertCoords & points, const CameraPointsTriangulationSettings & settings, const ProgressCallback & cb = {} );

/// same as above, but only valid points of the cloud participate in the triangulation (invalid points become invalid vertices)
[[nodiscard]] MRMESH_API Expected<Mesh> triangulateCameraPoints( const PointCloud & cloud, const CameraPointsTriangulationSettings & settings, const ProgressCallback & cb = {} );

/// same as above, but moves cloud points into the resulting mesh instead of copying them
[[nodiscard]] MRMESH_API Expected<Mesh> triangulateCameraPoints( PointCloud && cloud, const CameraPointsTriangulationSettings & settings, const ProgressCallback & cb = {} );

/// Reduces the depth noise of a mesh produced by triangulateCameraPoints (camera at the origin looking along +Z):
/// the depth (z) field is made smooth by interpolateScalarsSmoothly with given parameters (params.stabilizer > 0 keeps
/// every vertex attracted to its measured depth), and every vertex is moved along its viewing ray to the new depth,
/// so the projection of the mesh in the image plane and its absence of self-intersections are preserved
MRMESH_API void smoothCameraMeshDepth( Mesh & mesh, const InterpolateScalarsParams & params );

} //namespace MR
