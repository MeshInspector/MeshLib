#pragma once

#include "MRMeshFwd.h"
#include "MRMeshCollidePrecise.h"

MR_EXTERN_C_BEGIN

typedef MRVectorVarEdgeTri MRContinuousContour;

typedef struct MRContinuousContours MRContinuousContours;

MRMESHC_API MRContinuousContour mrContinuousContoursGet( const MRContinuousContours* contours, size_t index );

MRMESHC_API size_t mrContinuousContoursSize( const MRContinuousContours* contours );

MRMESHC_API void mrContinuousContoursFree( MRContinuousContours* contours );

// Combines individual intersections into ordered contours with the properties:
// a. left  of contours on mesh A is inside of mesh B,
// b. right of contours on mesh B is inside of mesh A,
// c. each intersected edge has origin inside meshes intersection and destination outside of it
MRMESHC_API MRContinuousContours* mrOrderIntersectionContours( const MRMeshTopology* topologyA, const MRMeshTopology* topologyB, const MRPreciseCollisionResult* intersections );

MR_EXTERN_C_END
