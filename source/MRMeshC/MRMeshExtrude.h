#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

// TODO: MakeDegenerateBandAroundRegionParams

/**
 * \brief Create a band of degenerate faces along the border of the specified region and the rest of the mesh
 * \details The function is useful for extruding the region without changing the existing faces and creating holes
 *
 * @param mesh - the target mesh
 * @param region - the region required to be separated by a band of degenerate faces
 */
MRMESHC_API void mrMakeDegenerateBandAroundRegion( MRMesh* mesh, const MRFaceBitSet* region );

MR_EXTERN_C_END
