#include "MRMeshBuilder.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBuilder.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_VECTOR( VertMap )

int mrMeshBuilderUniteCloseVertices( MRMesh* mesh_, float closeDist, bool uniteOnlyBd, MRVertMap* optionalVertOldToNew )
{
    ARG( mesh );
    VertMap vertOldToNew;
    auto res = MeshBuilder::uniteCloseVertices( mesh, closeDist, uniteOnlyBd, optionalVertOldToNew ? &vertOldToNew : nullptr );
    if ( optionalVertOldToNew )
    {
        optionalVertOldToNew->size = vertOldToNew.size();
        optionalVertOldToNew->data = new MRVertId[optionalVertOldToNew->size];
        std::copy( vertOldToNew.vec_.begin(), vertOldToNew.vec_.end(), (VertId*)optionalVertOldToNew->data );
    }
    return res;
}
