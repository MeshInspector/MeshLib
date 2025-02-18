#include "MRMeshExtrude.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshExtrude.h"

using namespace MR;

REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( Mesh )

void mrMakeDegenerateBandAroundRegion( MRMesh* mesh_, const MRFaceBitSet* region_ )
{
    ARG( mesh ); ARG( region );
    makeDegenerateBandAroundRegion( mesh, region );
}
