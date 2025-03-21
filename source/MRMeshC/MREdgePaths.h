#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// expands the region (of faces or vertices) on given value (in meters). returns false if callback also returns false
MRMESHC_API bool dilateRegionForFace( const MRMesh* mesh, MRFaceBitSet* region, float dilation, MRProgressCallback callback );
MRMESHC_API bool dilateRegionForVert( const MRMesh* mesh, MRVertBitSet* region, float dilation, MRProgressCallback callback );
MRMESHC_API bool dilateRegionForUndirectedEdge( const MRMesh* mesh, MRUndirectedEdgeBitSet* region, float dilation, MRProgressCallback callback );

/// shrinks the region (of faces or vertices) on given value (in meters). returns false if callback also returns false
MRMESHC_API bool erodeRegionForFace( const MRMesh* mesh, MRFaceBitSet* region, float dilation, MRProgressCallback callback );
MRMESHC_API bool erodeRegionForVert( const MRMesh* mesh, MRVertBitSet* region, float dilation, MRProgressCallback callback );
MRMESHC_API bool erodeRegionForUndirectedEdge( const MRMesh* mesh, MRUndirectedEdgeBitSet* region, float dilation, MRProgressCallback callback );

MR_EXTERN_C_END
