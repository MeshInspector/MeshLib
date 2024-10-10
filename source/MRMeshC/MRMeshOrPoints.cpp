#include "MRMeshOrPoints.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshOrPoints.h"

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( MeshOrPoints )
REGISTER_AUTO_CAST( MeshOrPointsXf )
REGISTER_AUTO_CAST( PointCloud )

MRMeshOrPoints* mrMeshOrPointsFromMesh( const MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW( MeshOrPoints( mesh ) );
}

MRMeshOrPoints* mrMeshOrPointsFromPointCloud( const MRPointCloud* pc_ )
{
    ARG( pc );
    RETURN_NEW( MeshOrPoints( pc ) );
}

void mrMeshOrPointsFree( MRMeshOrPoints* mp_ )
{
    ARG_PTR( mp );
    delete mp;
}

MRMeshOrPointsXf* mrMeshOrPointsXfNew( const MRMeshOrPoints* obj_, const MRAffineXf3f* xf_ )
{
    ARG( obj ); ARG( xf );
    RETURN_NEW( MeshOrPointsXf { obj, xf } );
}

MRMeshOrPointsXf* mrMeshOrPointsXfFromMesh( const MRMesh* mesh_, const MRAffineXf3f* xf_ )
{
    ARG( mesh ); ARG( xf );
    RETURN_NEW( MeshOrPointsXf { { mesh }, xf } );
}

MRMeshOrPointsXf* mrMeshOrPointsXfFromPointCloud( const MRPointCloud* pc_, const MRAffineXf3f* xf_ )
{
    ARG( pc ); ARG( xf );
    RETURN_NEW( MeshOrPointsXf { { pc }, xf } );
}

void mrMeshOrPointsXfFree( MRMeshOrPointsXf* mp_ )
{
    ARG_PTR( mp );
    delete mp;
}
