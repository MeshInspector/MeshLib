#pragma once
#include "MRMeshFillHole.h"

MR_EXTERN_C_BEGIN

typedef enum MRVertexMass
{
    /// all vertices have same mass=1
    MRVertexMassUnit,
    /// vertex mass depends on local geometry and proportional to the area of first-ring triangles
    MRVertexMassNeiArea
} MRVertexMass;

typedef enum MREdgeWeights
{  
    /// all edges have same weight=1
    MREdgeWeightsUnit,
    /// edge weight depends on local geometry and uses cotangent values
    MREdgeWeightsCotan
    /// cotangent edge weights and equation weights inversely proportional to square root of local area
    //MREdgeWeightsCotanWithAreaEqWeight => use MREdgeWeights::Cotan and MRVertexMass::NeiArea instead
} MREdgeWeights;

typedef struct MRFillHoleNicelyParams
{
    /// how to triangulate the hole, must be specified by the user
    MRFillHoleParams triangulateParams;

    /// If false then additional vertices are created inside the patch for best mesh quality
    bool triangulateOnly;

    /// in triangulateOnly = false mode, edges specified by this bit-set will never be flipped, but they can be split so it is updated during the operation
    MRUndirectedEdgeBitSet* notFlippable;

    /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
    float maxEdgeLen;

    /// Maximum number of edge splits allowed during subdivision
    int maxEdgeSplits;

    /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value (in radians)
    float maxAngleChangeAfterFlip;

    /// Whether to make patch over the hole smooth both inside and on its boundary with existed surface
    bool smoothCurvature;

    /// Additionally smooth 3 layers of vertices near hole boundary both inside and outside of the hole
    bool naturalSmooth;

    /// edge weighting scheme for smoothCurvature mode
    MREdgeWeights edgeWeights;

    /// vertex mass scheme for smoothCurvature mode
    MRVertexMass vmass;

} MRFillHoleNicelyParams;

MRMESHC_API MRFillHoleNicelyParams mrFillHoleNicelyParamsNew( void );

/// fills a hole in mesh specified by one of its edge,
/// optionally subdivides new patch on smaller triangles,
/// optionally make smooth connection with existing triangles outside the hole
/// \return triangles of the patch
MRMESHC_API MRFaceBitSet* mrFillHoleNicely( MRMesh* mesh, MREdgeId holeEdge_, const MRFillHoleNicelyParams* params );

MR_EXTERN_C_END
