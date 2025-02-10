#pragma once

#include "MRMeshFwd.h"
#include "MRRelaxParams.h"

MR_EXTERN_C_BEGIN

// TODO: MeshRelaxParams

/// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESHC_API bool mrRelax( MRMesh* mesh, const MRRelaxParams* params, MRProgressCallback* cb );

MR_EXTERN_C_END
