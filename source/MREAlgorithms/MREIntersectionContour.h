#pragma once
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRMeshCollidePrecise.h"

namespace MRE
{

struct VariableEdgeTri : MR::EdgeTri
{
    bool isEdgeATriB{false};
};

using ContinuousContour = std::vector<VariableEdgeTri>;
using ContinuousContours = std::vector<ContinuousContour>;

// Combines individual intersections into ordered contours with the properties:
// a. left  of contours on mesh A is inside of mesh B,
// b. right of contours on mesh B is inside of mesh A,
// c. each intersected edge has origin inside meshes intersection and destination outside of it
MREALGORITHMS_API ContinuousContours orderIntersectionContours( const MR::MeshTopology& topologyA, const MR::MeshTopology& topologyB, const MR::PreciseCollisionResult& intersections );

// Detects contours that fully lay inside one triangle
// returns they indices in contours
MREALGORITHMS_API std::vector<int> detectLoneContours( const ContinuousContours& contours );

// Removes contours that fully lay inside one triangle from the contours
MREALGORITHMS_API void removeLoneContours( ContinuousContours& contours );

}
