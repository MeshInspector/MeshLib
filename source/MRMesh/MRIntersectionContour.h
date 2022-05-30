#pragma once

#include "MRMeshCollidePrecise.h"

namespace MR
{

struct VariableEdgeTri : EdgeTri
{
    bool isEdgeATriB{false};
};

using ContinuousContour = std::vector<VariableEdgeTri>;
using ContinuousContours = std::vector<ContinuousContour>;

// Combines individual intersections into ordered contours with the properties:
// a. left  of contours on mesh A is inside of mesh B,
// b. right of contours on mesh B is inside of mesh A,
// c. each intersected edge has origin inside meshes intersection and destination outside of it
MRMESH_API ContinuousContours orderIntersectionContours( const MeshTopology& topologyA, const MeshTopology& topologyB, const PreciseCollisionResult& intersections );

// Detects contours that fully lay inside one triangle
// returns they indices in contours
MRMESH_API std::vector<int> detectLoneContours( const ContinuousContours& contours );

// Removes contours that fully lay inside one triangle from the contours
MRMESH_API void removeLoneContours( ContinuousContours& contours );

}
