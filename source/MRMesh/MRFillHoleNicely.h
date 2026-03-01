#pragma once

#include "MRMeshFwd.h"
#include "MRMeshFillHole.h"
#include "MREnums.h"

namespace MR
{

struct SubdivideFillingSettings
{
    /// in triangulateOnly = false mode, edges specified by this bit-set will never be flipped, but they can be split so it is updated during the operation
    UndirectedEdgeBitSet* notFlippable = nullptr;

    /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
    float maxEdgeLen = 0;

    /// Maximum number of edge splits allowed during subdivision
    int maxEdgeSplits = 1000;

    /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value (in radians)
    float maxAngleChangeAfterFlip = 30 * PI_F / 180.0f;

    /// (If this is set) this function is called in subdivision each time edge (e) is going to split, if it returns false then this split will be skipped
    std::function<bool( EdgeId e )> beforeEdgeSplit;

    /// (If this is set) this function is called in subdivision each time edge (e) is split into (e1->e), but before the ring is made Delone
    std::function<void( EdgeId e1, EdgeId e )> onEdgeSplit;
};

struct SmoothFillingSettings
{
    /// Additionally smooth 3 layers of vertices near hole boundary both inside and outside of the hole
    bool naturalSmooth = false;

    /// edge weighting scheme for smoothCurvature mode
    EdgeWeights edgeWeights = EdgeWeights::Cotan;

    /// vertex mass scheme for smoothCurvature mode
    VertexMass vmass = VertexMass::Unit;
};

struct OutAttributesFillingSettings
{
    /// optional uv-coordinates of vertices; if provided then elements corresponding to new vertices will be added there
    VertUVCoords* uvCoords = {};

    /// optional colors of vertices; if provided then elements corresponding to new vertices will be added there
    VertColors* colorMap = {};

    /// optional colors of faces; if provided then elements corresponding to new faces will be added there
    FaceColors* faceColors = {};
};

struct FillHoleNicelySettings
{
    /// how to triangulate the hole, must be specified by the user
    FillHoleParams triangulateParams;

    /// If false then additional vertices are created inside the patch for best mesh quality
    bool triangulateOnly = false;

    /// if `triangulateOnly` is false - this settings are used to subdivide new filling
    SubdivideFillingSettings subdivideSettings;

    /// Whether to make patch over the hole smooth both inside and on its boundary with existed surface
    bool smoothCurvature = true;

    /// if `smoothCurvature` is true and `triangulateOnly is false - these settings are used to smooth new filling
    SmoothFillingSettings smoothSeettings;

    /// structure with optional output attributes
    OutAttributesFillingSettings outAttributes;
};

/// fills a hole in mesh specified by one of its edge,
/// optionally subdivides new patch on smaller triangles,
/// optionally make smooth connection with existing triangles outside the hole
/// \return triangles of the patch
MRMESH_API FaceBitSet fillHoleNicely( Mesh & mesh,
    EdgeId holeEdge, ///< left of this edge must not have a face and it will be filled
    const FillHoleNicelySettings & settings );

struct StitchHolesNicelySettings
{
    /// how to triangulate the cylinder between holes, must be specified by the user
    StitchHolesParams triangulateParams;

    /// If false then additional vertices are created inside the patch for best mesh quality
    bool triangulateOnly = false;

    /// if `triangulateOnly` is false - this settings are used to subdivide new filling
    SubdivideFillingSettings subdivideSettings;

    /// Whether to make patch over the hole smooth both inside and on its boundary with existed surface
    bool smoothCurvature = true;

    /// if `smoothCurvature` is true and `triangulateOnly is false - these settings are used to smooth new filling
    SmoothFillingSettings smoothSeettings;

    /// structure with optional output attributes
    OutAttributesFillingSettings outAttributes;
};

/// stitches two holes building cylinder between them, each hole is specified by one of its edge,
/// optionally subdivides new patch on smaller triangles,
/// optionally make smooth connection with existing triangles outside the hole
/// \return triangles of the patch
MRMESH_API FaceBitSet stitchHolesNicely( Mesh& mesh, 
    EdgeId hole0Edge, ///< left of this edge must not have a face and it will be filled
    EdgeId hole1Edge, ///< left of this edge must not have a face and it will be filled
    const StitchHolesNicelySettings& settings );

} //namespace MR
