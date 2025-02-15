#include "MREdgePaths.h"

#include "detail/TypeCast.h"

#include "MRMesh/MREdgePaths.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( UndirectedEdgeBitSet )
REGISTER_AUTO_CAST( VertBitSet )

bool dilateRegionForFace( const MRMesh* mesh_, MRFaceBitSet* region_, float dilation, MRProgressCallback callback )
{
    ARG( mesh ); ARG( region );
    return dilateRegion( mesh, region, dilation, callback );
}

bool dilateRegionForVert( const MRMesh* mesh_, MRVertBitSet* region_, float dilation, MRProgressCallback callback )
{
    ARG( mesh ); ARG( region );
    return dilateRegion( mesh, region, dilation, callback );
}

bool dilateRegionForUndirectedEdge( const MRMesh* mesh_, MRUndirectedEdgeBitSet* region_, float dilation, MRProgressCallback callback )
{
    ARG( mesh ); ARG( region );
    return dilateRegion( mesh, region, dilation, callback );
}

bool erodeRegionForFace( const MRMesh* mesh_, MRFaceBitSet* region_, float dilation, MRProgressCallback callback )
{
    ARG( mesh ); ARG( region );
    return erodeRegion( mesh, region, dilation, callback );
}

bool erodeRegionForVert( const MRMesh* mesh_, MRVertBitSet* region_, float dilation, MRProgressCallback callback )
{
    ARG( mesh ); ARG( region );
    return erodeRegion( mesh, region, dilation, callback );
}

bool erodeRegionForUndirectedEdge( const MRMesh* mesh_, MRUndirectedEdgeBitSet* region_, float dilation, MRProgressCallback callback )
{
    ARG( mesh ); ARG( region );
    return erodeRegion( mesh, region, dilation, callback );
}
