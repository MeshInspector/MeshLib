#include "MRPointCloud.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRBox.h"
#include "MRMesh/MRPointCloud.h"

#include <span>

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( Box3f )
REGISTER_AUTO_CAST( PointCloud )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST( VertBitSet )
REGISTER_AUTO_CAST( VertId )

MRPointCloud* mrPointCloudNew( void )
{
    RETURN_NEW( PointCloud() );
}

MRPointCloud* mrPointCloudFromPoints( const MRVector3f* points_, size_t pointsNum )
{
    std::span points { auto_cast( points_ ), pointsNum };

    PointCloud res;
    res.points = { points.begin(), points.end() };
    res.validPoints.resize( pointsNum, true );
    RETURN_NEW( std::move( res ) );
}

const MRVector3f* mrPointCloudPoints( const MRPointCloud* pc_ )
{
    ARG( pc );
    RETURN( pc.points.data() );
}

MRVector3f* mrPointCloudPointsRef( MRPointCloud* pc_ )
{
    ARG( pc );
    RETURN( pc.points.data() );
}

size_t mrPointCloudPointsNum( const MRPointCloud* pc_ )
{
    ARG( pc );
    return pc.points.size();
}

const MRVector3f* mrPointCloudNormals( const MRPointCloud* pc_ )
{
    ARG( pc );
    RETURN( pc.normals.data() );
}

size_t mrPointCloudNormalsNum( const MRPointCloud* pc_ )
{
    ARG( pc );
    return pc.normals.size();
}

const MRVertBitSet* mrPointCloudValidPoints( const MRPointCloud* pc_ )
{
    ARG( pc );
    RETURN( &pc.validPoints );
}

MRBox3f mrPointCloudComputeBoundingBox( const MRPointCloud* pc_, const MRAffineXf3f* toWorld_ )
{
    ARG( pc ); ARG_PTR( toWorld );
    RETURN( pc.computeBoundingBox( toWorld ) );
}

MRVertId mrPointCloudAddPoint( MRPointCloud* pc_, const MRVector3f* point_ )
{
    ARG( pc ); ARG( point );
    RETURN( pc.addPoint( point ) );
}

MRVertId mrPointCloudAddPointWithNormal( MRPointCloud* pc_, const MRVector3f* point_, const MRVector3f* normal_ )
{
    ARG( pc ); ARG( point ); ARG( normal );
    RETURN( pc.addPoint( point, normal ) );
}

void mrPointCloudFree( MRPointCloud* pc_ )
{
    ARG_PTR( pc );
    delete pc;
}

void mrPointCloudInvalidateCaches( MRPointCloud* pc_ )
{
    ARG( pc );
    pc.invalidateCaches();
}

void mrPointCloudSetValidPoints( MRPointCloud* pc_, const MRVertBitSet* validPoints_ )
{
    ARG( pc ); ARG( validPoints );
    pc.validPoints = validPoints;
}
