#include "MRMeshFixer.h"
#include "MRBitSet.h"
#include "MRMesh/MRMeshFixer.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRMeshFixer.h"
#include "detail/TypeCast.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( FaceBitSet )

MRFaceBitSet* mrFindHoleComplicatingFaces( MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW( findHoleComplicatingFaces( mesh ) );
}
