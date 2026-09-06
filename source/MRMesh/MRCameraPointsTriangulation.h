#pragma once

#include "MRMeshFwd.h"
#include "MRMatrix3.h"
#include "MRMeshDelone.h"
#include "MRConstants.h"
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

    /// if not null, receives the result of findSmallestCloseVertices on the image-plane positions:
    /// each point is mapped to the point it was merged into (or to itself); left unchanged if weldPixels <= 0
    VertMap * outSmallestMap = nullptr;

    /// the number of iterations of makeDeloneEdgeFlips at the end to improve the triangulation in 3D,
    /// since Delaunay property in the image plane is not the same as in space; zero disables the flips
    int numDeloneIters = 1;

    /// parameters of these flips; the default limit on dihedral angle change keeps the mesh free of self-intersections,
    /// which unlimited flips of the quadrangles non-convex in the image plane otherwise introduce
    DeloneSettings deloneSettings{ .maxAngleChange = PI_F / 3 };
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

} //namespace MR
