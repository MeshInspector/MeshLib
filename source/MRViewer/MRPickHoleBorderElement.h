#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRMeshEdgePoint.h"
#include "exports.h"

namespace MR
{

struct HoleEdgePoint
{
    int holeIdx{ -1 }; ///< hole index
    MeshEdgePoint edgePoint;
};

/// find closest to mouse edge from hole borders
/// \mousePos mouse position in screen
/// \accuracySq square of accuracy in pixels
/// \param accuracy,cornerAccuracy maximum distances from mouse position to line / corner in viewport (screen) space
/// \return pair closest edge and according index of hole in holeRepresentativeEdges
MRVIEWER_API HoleEdgePoint findClosestToMouseHoleEdge( const Vector2i& mousePos, const std::shared_ptr<ObjectMesh>& objMesh,
                                                       const std::vector<EdgeId>& holeRepresentativeEdges,
                                                       float accuracy = 5.5f, bool attractToVert = false, float cornerAccuracy = 10.5f );

}
