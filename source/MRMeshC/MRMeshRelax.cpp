#include "MRMeshRelax.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshRelax.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( RelaxParams )

bool mrRelax( MRMesh* mesh_, const MRRelaxParams* params_, MRProgressCallback* cb )
{
    ARG( mesh ); ARG_PTR( params );
    return relax( mesh, MeshRelaxParams { params ? *params : RelaxParams{} }, cb ? *cb : ProgressCallback{} );
}

bool mrRelaxKeepVolume( MRMesh* mesh_, const MRRelaxParams* params_, MRProgressCallback* cb )
{
    ARG( mesh ); ARG_PTR( params );
    return relaxKeepVolume( mesh, MeshRelaxParams{ params ? *params : RelaxParams{} }, cb ? *cb : ProgressCallback{} );
}
