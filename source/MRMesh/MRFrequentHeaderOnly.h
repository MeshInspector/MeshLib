#pragma once

// This header aggregates MRMesh public headers that
// 1) never use the MRMESH_API macro (directly or via sub-includes), i.e. they are effectively header-only, and
// 2) are included (directly or indirectly) by more than 20 of MeshLib's .cpp files.
// The number in the comment after each include is that count of including .cpp files.
// Such headers are good candidates for precompiled-header baking.

#include "MRMacros.h" // 647
#include "MRCanonicalTypedefs.h" // 646
#include "MRMeshFwd.h" // 646
#include "config.h" // 646
#include "MRUnsigned.h" // 560
#include "MRConstants.h" // 543
#include "MRVector3.h" // 542
#include "MRId.h" // 497
#include "MRphmap.h" // 469
#include "MRResizeNoInit.h" // 462
#include "MRVector.h" // 462
#include "MRVector2.h" // 461
#include "MRExpected.h" // 452
#include "MRProgressCallback.h" // 433
#include "MRMatrix2.h" // 381
#include "MRMatrix3.h" // 377
#include "MRAffineXf.h" // 371
#include "MRVectorTraits.h" // 368
#include "MRSegmPoint.h" // 358
#include "MRAffineXf3.h" // 356
#include "MRMeshPart.h" // 312
#include "MRBox.h" // 306
#include "MRTriPoint.h" // 304
#include "MRVector4.h" // 263
#include "MRWriter.h" // 257
#include "MRMeshBuilderTypes.h" // 256
#include "MRPlane3.h" // 212
#include "MRViewportId.h" // 211
#include "MRHeapBytes.h" // 208
#include "MRFlagOperators.h" // 203
#include "MRRenderModelParameters.h" // 195
#include "MRSignal.h" // 192
#include "MRFunctional.h" // 168
#include "MRParallel.h" // 168
#include "MRViewportProperty.h" // 148
#include "MRLineSegm.h" // 141
#include "MRUniquePtr.h" // 127
#include "MRXfBasedCache.h" // 88
#include "MRFinally.h" // 81
#include "MRIteratorRange.h" // 78
#include "MRPointCloudPart.h" // 72
#include "MRMatrix4.h" // 71
#include "MRBuffer.h" // 68
#include "MRNoDefInit.h" // 68
#include "MRQuaternion.h" // 68
#include "MRCloudPartMapping.h" // 65
#include "MRLine.h" // 63
#include "MRLine3.h" // 56
#include "MRMeshLoadSettings.h" // 53
#include "MROnInit.h" // 53
#include "MRLinesLoadSettings.h" // 47
#include "MRPointsLoadSettings.h" // 46
#include "MRAABBTreeNode.h" // 36
#include "MRSymMatrix3.h" // 34
#include "MRTriMath.h" // 26
#include "MRHoleFillPlan.h" // 25
