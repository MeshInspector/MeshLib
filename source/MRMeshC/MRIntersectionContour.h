#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRMeshCollidePrecise.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

/// ...
typedef struct MRVariableEdgeTri
{
    MREdgeId edge;
    MRFaceId tri;
    bool isEdgeATriB;
}
MRVariableEdgeTri;

MR_VECTOR_LIKE_DECL( ContinuousContour, VariableEdgeTri )

/// ...
typedef struct MRContinuousContours MRContinuousContours;

/// ...
MRMESHC_API const MRContinuousContour mrContinuousContoursGet( const MRContinuousContours* contours, size_t index );

/// ...
MRMESHC_API size_t mrContinuousContoursSize( const MRContinuousContours* contours );

/// ...
MRMESHC_API void mrContinuousContoursFree( MRContinuousContours* contours );

/// ...
MRMESHC_API MRContinuousContours* mrOrderIntersectionContours( const MRMeshTopology* topologyA, const MRMeshTopology* topologyB, const MRPreciseCollisionResult* intersections );

MR_EXTERN_C_END
