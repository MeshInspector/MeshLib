#pragma once

#include "MRMeshFwd.h"
#include "MRMeshPart.h"
MR_EXTERN_C_BEGIN
/// returns all faces that complicate one of mesh holes;
/// hole is complicated if it passes via one vertex more than once;
/// deleting such faces simplifies the holes and makes them easier to fill
MRMESHC_API MRFaceBitSet* mrFindHoleComplicatingFaces( MRMesh* mesh );

MRMESHC_API MRFaceBitSet* mrFindDegenerateFaces( const MRMeshPart* mp, float criticalAspectRatio, MRProgressCallback cb, MRString** errorString );

MRMESHC_API MRUndirectedEdgeBitSet* mrFindShortEdges( const MRMeshPart* mp, float criticalLength, MRProgressCallback cb, MRString** errorString );

MR_EXTERN_C_END
