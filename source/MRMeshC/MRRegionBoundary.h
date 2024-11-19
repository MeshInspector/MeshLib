#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRMeshTopology.h"

MR_EXTERN_C_BEGIN

/// returns closed loop of region boundary starting from given region boundary edge (region faces on the right, and not-region faces or holes on the left);
/// if more than two boundary edges connect in one vertex, then the function makes the most abrupt turn to left
MRMESHC_API MREdgeLoop* mrTrackRightBoundaryLoop( const MRMeshTopology* topology, MREdgeId e0, const MRFaceBitSet* region );

typedef struct MREdgeLoops MREdgeLoops;

MRMESHC_API const MREdgeLoop mrEdgeLoopsGet( const MREdgeLoops* loops, size_t index );

MRMESHC_API size_t mrEdgeLoopsSize( const MREdgeLoops* loops );

MRMESHC_API void mrEdgeLoopsFree( MREdgeLoops* loops );

/// returns all region boundary loops;
/// every loop has region faces on the right, and not-region faces or holes on the left
MRMESHC_API MREdgeLoops* mrFindRightBoundary( const MRMeshTopology* topology, const MRFaceBitSet* region );

MRMESHC_API MRFaceBitSet* mrGetIncidentFacesFromVerts( const MRMeshTopology* topology, const MRVertBitSet* region );

MRMESHC_API MRFaceBitSet* mrGetIncidentFacesFromEdges( const MRMeshTopology* topology, const MRUndirectedEdgeBitSet* region );

MR_EXTERN_C_END
