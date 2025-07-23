#pragma once

#include "MRMeshFwd.h"
#include "MRRelaxParams.h"

MR_EXTERN_C_BEGIN

// TODO: MeshRelaxParams

/// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESHC_API bool mrRelax( MRMesh* mesh, const MRRelaxParams* params, MRProgressCallback* cb );

/// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified ) \n
/// do not really keeps volume but tries hard
/// \return true if the operation completed successfully, and false if it was interrupted by the progress callback.
MRMESHC_API bool mrRelaxKeepVolume( MRMesh* mesh, const MRRelaxParams* params, MRProgressCallback* cb );

MR_EXTERN_C_END
