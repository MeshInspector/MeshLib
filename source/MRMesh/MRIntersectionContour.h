#pragma once

#include "MRMeshCollidePrecise.h"
#include "MRPch/MRBindingMacros.h"

namespace MR
{

struct OneMeshContour;
using OneMeshContours = std::vector<OneMeshContour>;

using ContinuousContour = std::vector<VarEdgeTri>;
using ContinuousContours = std::vector<ContinuousContour>;

/// Combines unordered input intersections (and flips orientation of intersected edges from mesh B) into ordered oriented contours with the properties:
/// 1. Each contour is
///    a. either closed (then its first and last elements are equal),
///    b. or open (then its first and last intersected edges are boundary edges).
/// 2. Next intersection in a contour is located to the left of the current intersected edge:
///    a. if the current and next intersected triangles are the same, then next intersected edge is either next( curr.edge ) or prev( curr.edge.sym() ).sym(),
///    b. otherwise next intersected triangle is left( curr.edge ) and next intersected edge is one of the edges having the current intersected triangle to the right.
/// 3. Orientation of intersected edges in each pair of (intersected edge, intersected triangle):
///    a. the intersected edge of mesh A is directed from negative half-space of the intersected triangle from mesh B to its positive half-space,
///    b. the intersected edge of mesh B is directed from positive half-space of the intersected triangle from mesh A to its negative half-space.
/// 4. Orientation of contours:
///    a. left  of contours on mesh A is inside of mesh B (consequence of 3a),
///    b. right of contours on mesh B is inside of mesh A (consequence of 3b).
MRMESH_API ContinuousContours orderIntersectionContours( const MeshTopology& topologyA, const MeshTopology& topologyB, const PreciseCollisionResult& intersections );

/// Combines unordered input self-intersections (and flips orientation of some intersected edges) into ordered oriented contours with the properties:
/// 1. Each contour is
///    a. either closed (then its first and last elements are equal),
///    b. or open if terminal intersection is on mesh boundary or if self-intersection terminates in a vertex.
/// 2. Next intersection in a contour is located to the left of the current intersected edge:
///    a. if the current and next intersected triangles are the same, then next intersected edge is either next( curr.edge ) or prev( curr.edge.sym() ).sym(),
///    b. otherwise next intersected triangle is left( curr.edge ) and next intersected edge is one of the edges having the current intersected triangle to the right.
/// 3. Orientation of intersected edges in each pair of (intersected edge, intersected triangle):
///    a. isEdgeATriB() = true:  the intersected edge is directed from negative half-space of the intersected triangle to its positive half-space,
///    b. isEdgeATriB() = false: the intersected edge is directed from positive half-space of the intersected triangle to its negative half-space.
/// 4. Contours [2*i] and [2*i+1]
///    a. have equal lengths and pass via the same intersections but in opposite order,
///    b. each intersection is present in two contours with different values of isEdgeATriB() flag, and opposite directions of the intersected edge.
/// 5. Orientation of contours:
///    a. first element of even (0,2,...) contours has isEdgeATriB() = true, left of even contours goes inside (consequence of 3a),
///    b. first element of odd (1,3,...) contours has isEdgeATriB() = false, right of odd contours goes inside (consequence of 3b).
MRMESH_API ContinuousContours orderSelfIntersectionContours( const MeshTopology& topology, const std::vector<EdgeTri>& intersections );

/// extracts coordinates from two meshes intersection contours
[[deprecated( "Use getOneMeshIntersectionContours")]] MRMESH_API MR_BIND_IGNORE Contours3f extractIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& orientedContours,
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
