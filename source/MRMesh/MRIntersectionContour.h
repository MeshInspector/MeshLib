#pragma once

#include "MRMeshCollidePrecise.h"

namespace MR
{

struct OneMeshContour;
using OneMeshContours = std::vector<OneMeshContour>;

using ContinuousContour = std::vector<VarEdgeTri>;
using ContinuousContours = std::vector<ContinuousContour>;

/// Combines individual intersections into ordered contours with the properties:
/// a. left  of contours on mesh A is inside of mesh B,
/// b. right of contours on mesh B is inside of mesh A,
/// c. each intersected edge has origin inside meshes intersection and destination outside of it
MRMESH_API ContinuousContours orderIntersectionContours( const MeshTopology& topologyA, const MeshTopology& topologyB, const PreciseCollisionResult& intersections );

/// Combines individual self intersections into ordered contours with the properties:
/// a. left  of contours on mesh is inside
/// b. each intersected edge has origin inside meshes intersection and destination outside of it
MRMESH_API ContinuousContours orderSelfIntersectionContours( const MeshTopology& topology, const std::vector<EdgeTri>& intersections );

/// extracts coordinates from two meshes intersection contours
MRMESH_API Contours3f extractIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& orientedContours, 
const CoordinateConverters& converters, const AffineXf3f* rigidB2A = nullptr );

/// returns true if contour is closed
MRMESH_API bool isClosed( const ContinuousContour& contour );

/// Detects contours that fully lay inside one triangle
/// if `ignoreOpen` then do not mark non-closed contours as lone, even if they really are
/// returns they indices in contours
MRMESH_API std::vector<int> detectLoneContours( const ContinuousContours& contours, bool ignoreOpen = false );

/// Removes contours with zero area (do not remove if contour is handle on topology)
/// edgesTopology - topology on which contours are represented with edges
/// faceContours - lone contours represented by faces (all intersections are in same mesh A face)
/// edgeContours - lone contours represented by edges (all intersections are in mesh B edges, edgesTopology: meshB.topology)
MRMESH_API void removeLoneDegeneratedContours( const MeshTopology& edgesTopology, 
    OneMeshContours& faceContours, OneMeshContours& edgeContours );

/// Removes contours that fully lay inside one triangle from the contours
/// if `ignoreOpen` then do not consider non-closed contours as lone, even if they really are
MRMESH_API void removeLoneContours( ContinuousContours& contours, bool ignoreOpen = false );

}
