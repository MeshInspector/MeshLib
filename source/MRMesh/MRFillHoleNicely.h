#pragma once

#include "MRMeshFwd.h"
#include "MRMeshFillHole.h"
#include "MREnums.h"

namespace MR
{

struct FillHoleNicelySettings
{
    /// how to triangulate the hole, must be specified by the user
    FillHoleParams triangulateParams;

    /// If false then additional vertices are created inside the patch for best mesh quality
    bool triangulateOnly = false;

    /// in triangulateOnly = false mode, edges specified by this bit-set will never be flipped, but they can be split so it is updated during the operation
    UndirectedEdgeBitSet* notFlippable = nullptr;

    /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
    float maxEdgeLen = 0;

    /// Maximum number of edge splits allowed during subdivision
    int maxEdgeSplits = 1000;

    /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value (in radians)
    float maxAngleChangeAfterFlip = 30 * PI_F / 180.0f;

    /// Whether to make patch over the hole smooth both inside and on its boundary with existed surface
    bool smoothCurvature = true;

    /// Additionally smooth 3 layers of vertices near hole boundary both inside and outside of the hole
    bool naturalSmooth = false;

    /// (If this is set) this function is called in subdivision each time edge (e) is going to split, if it returns false then this split will be skipped
    std::function<bool( EdgeId e )> beforeEdgeSplit;

    /// (If this is set) this function is called in subdivision each time edge (e) is split into (e1->e), but before the ring is made Delone
    std::function<void( EdgeId e1, EdgeId e )> onEdgeSplit;

    /// edge weighting scheme for smoothCurvature mode
    EdgeWeights edgeWeights = EdgeWeights::Cotan;

    /// vertex mass scheme for smoothCurvature mode
    VertexMass vmass = VertexMass::Unit;

    /// optional uv-coordinates of vertices; if provided then elements corresponding to new vertices will be added there
    VertUVCoords * uvCoords = {};

    /// optional colors of vertices; if provided then elements corresponding to new vertices will be added there
    VertColors * colorMap = {};

    /// optional colors of faces; if provided then elements corresponding to new faces will be added there
    FaceColors * faceColors = {};
};

/// fills a hole in mesh specified by one of its edge,
/// optionally subdivides new patch on smaller triangles,
/// optionally make smooth connection with existing triangles outside the hole
/// \return triangles of the patch
MRMESH_API FaceBitSet fillHoleNicely( Mesh & mesh,
    EdgeId holeEdge, ///< left of this edge must not have a face and it will be filled
    const FillHoleNicelySettings & settings );

} //namespace MR
