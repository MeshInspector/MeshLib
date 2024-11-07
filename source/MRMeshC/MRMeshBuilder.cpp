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
    if ( !optionalVertOldToNew )
        return MeshBuilder::uniteCloseVertices( mesh, closeDist, uniteOnlyBd, nullptr );

    VertMap vertOldToNew;
    auto res =  MeshBuilder::uniteCloseVertices( mesh, closeDist, uniteOnlyBd, &vertOldToNew );
    optionalVertOldToNew->size = vertOldToNew.size();
    optionalVertOldToNew->data = new MRVertId[optionalVertOldToNew->size];
    std::copy( vertOldToNew.vec_.begin(), vertOldToNew.vec_.end(), (VertId*)optionalVertOldToNew->data );    
    return res;
}

MRVertMap mrMeshBuilderVertMapNew( void )
{
    MRVertMap res;
    res.size = 0;
    res.data = nullptr;
    res.reserved1 = nullptr;
    return res;
}

void mrMeshBuilderVertMapFree( MRVertMap* vertOldToNew )
{
    if ( !vertOldToNew )
        return;

    delete[] vertOldToNew->data;
    vertOldToNew->data = nullptr;
    vertOldToNew->size = 0;
}