#pragma once

// Lean precompiled-header payload for GCC. Unlike MSVC, GCC loads the whole PCH
// image into every TU, so this caches only the high-fan-in but lightweight MeshLib
// headers (ids, vectors/matrices, transforms, geometry primitives) rather than the
// heavy MRMesh.h: they are included in nearly every TU yet add little to the image.
// They only fit the shared PCH because <target>_EXPORTS is no longer defined on
// non-MSVC (avoiding -Winvalid-pch).

#include "../MRMesh/MRMeshFwd.h"
#include "../MRMesh/MRUnsigned.h"
#include "../MRMesh/MRConstants.h"
#include "../MRMesh/MRId.h"
#include "../MRMesh/MRphmap.h"
#include "../MRMesh/MRResizeNoInit.h"
#include "../MRMesh/MRVector.h"
#include "../MRMesh/MRVector2.h"
#include "../MRMesh/MRVector3.h"
#include "../MRMesh/MRVector4.h"
#include "../MRMesh/MRExpected.h"
#include "../MRMesh/MRProgressCallback.h"
#include "../MRMesh/MRMatrix2.h"
#include "../MRMesh/MRMatrix3.h"
#include "../MRMesh/MRMatrix4.h"
#include "../MRMesh/MRAffineXf3.h"
#include "../MRMesh/MRAffineXf.h"
#include "../MRMesh/MRVectorTraits.h"
#include "../MRMesh/MRSegmPoint.h"
#include "../MRMesh/MRBox.h"
#include "../MRMesh/MRTriPoint.h"
#include "../MRMesh/MRBuffer.h"
#include "../MRMesh/MRChrono.h"
#include "../MRMesh/MRBitSet.h"
#include "../MRMesh/MRTimer.h"
#include "../MRMesh/MRFile.h"

// MRViewerFwd.h is likewise forward-declaration-only (config.h, exports.h,
// MRMeshFwd.h, <functional>) and included across MRViewer and its consumers.
#include "../MRViewer/MRViewerFwd.h"
