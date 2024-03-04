#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MREdgePoint.h"
#include "exports.h"

namespace MR
{

struct HoleEdgePoint
{
    int holeIdx{ -1 }; ///< hole index
    MeshEdgePoint edgePoint;
};

/// find closest to mouse edge from hole borders
/// \param mousePos mouse position in screen
/// \param accuracy, cornerAccuracy maximum distances from mouse position to line / corner in viewport (screen) space
/// \return pair closest edge and according index of hole in holeRepresentativeEdges
MRVIEWER_API HoleEdgePoint findClosestToMouseHoleEdge( const Vector2i& mousePos, const std::shared_ptr<ObjectMeshHolder>& objMesh,
                                                       const std::vector<EdgeId>& holeRepresentativeEdges,
                                                       float accuracy = 5.5f, bool attractToVert = false, float cornerAccuracy = 10.5f );

/// find closest to mouse edge from polylines
/// \param mousePos mouse position in screen
/// \param accuracy maximum distances from mouse position to line in viewport (screen) space
/// \return pair closest edge and according index of polyline
MRVIEWER_API HoleEdgePoint findClosestToMouseEdge( const Vector2i& mousePos, const std::vector<std::shared_ptr<ObjectLinesHolder>>& objsLines,
                                                   float accuracy = 5.5f );
inline HoleEdgePoint findClosestToMouseEdge( const Vector2i& mousePos, const std::vector<std::shared_ptr<ObjectLines>>& objsLines,
                                      float accuracy = 5.5f )
{
    std::vector<std::shared_ptr<ObjectLinesHolder>> objs(objsLines.size());
    for ( int i = 0; i < objsLines.size(); ++i )
        objs[i] = std::dynamic_pointer_cast<ObjectLinesHolder>( objsLines[i] );
    return findClosestToMouseEdge( mousePos, objs, accuracy );
}

}
