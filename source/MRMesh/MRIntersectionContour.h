#pragma once

#include "MRMeshCollidePrecise.h"

namespace MR
{

struct OneMeshContour;
using OneMeshContours = std::vector<OneMeshContour>;

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
// returns their indices in contours
MRMESH_API BitSet detectLoneContours( const ContinuousContours& contours );

// Detects contours degenerated to single point among contourIds
// faceContours - lone contours represented by faces (all intersections are in same mesh A face)
// returns their indices in contours
MRMESH_API BitSet detectSingularContours( const OneMeshContours& faceContours, const BitSet& contourIds );

// Detects contours with zero area among contourIds
// faceContours - lone contours represented by faces (all intersections are in same mesh A face)
// returns their indices in contours
MRMESH_API BitSet detectDegeneratedContours( const OneMeshContours& faceContours, const BitSet& contourIds );

// Detects contours which are handles on surface among contourIds
// tpA - topology on mesh A
// tpB - topology on mesh B
MRMESH_API BitSet detectNonTrivialContours( const MeshTopology& tpA, const MeshTopology& tpB, const ContinuousContours& contours, const BitSet& contourIds );

// Removes contours by contourIds from faceContours and edgeContours
// faceContours - lone contours represented by faces (all intersections are in same mesh A face)
MRMESH_API void removeContours( OneMeshContours& contours, const BitSet& contourIds );
// Removes contours by contourIds from contours
MRMESH_API void removeContours( ContinuousContours& contours, const BitSet& contourIds );

}
