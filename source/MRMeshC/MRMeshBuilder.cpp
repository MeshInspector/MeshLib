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
    vector_wrapper<VertId>* wrapper = ( vector_wrapper<VertId>* )( optionalVertOldToNew );
    VertMap* vmap = nullptr;
    if ( wrapper )
    {
        vmap = ( VertMap* )( &( std::vector<VertId>&)(* wrapper ));
    }

    auto res =  MeshBuilder::uniteCloseVertices( mesh, closeDist, uniteOnlyBd, vmap );
    if ( optionalVertOldToNew )
    {
        mrVertMapInvalidate( optionalVertOldToNew );
    }
    return res;
        
}
