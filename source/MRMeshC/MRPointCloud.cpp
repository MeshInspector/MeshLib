#include "MRPointCloud.h"

#include "MRMesh/MRBox.h"
#include "MRMesh/MRPointCloud.h"

#include <span>

using namespace MR;

MRPointCloud* mrPointCloudNew( void )
{
    return reinterpret_cast<MRPointCloud*>( new PointCloud );
}

MRPointCloud* mrPointCloudFromPoints( const MRVector3f* points_, size_t pointsNum )
{
    std::span points { reinterpret_cast<const Vector3f*>( points_ ), pointsNum };

    PointCloud res;
    res.points = { points.begin(), points.end() };
    return reinterpret_cast<MRPointCloud*>( new PointCloud( std::move( res ) ) );
}

const MRVector3f* mrPointCloudPoints( const MRPointCloud* pc_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    return reinterpret_cast<const MRVector3f*>( pc.points.data() );
}

MRVector3f* mrPointCloudPointsRef( MRPointCloud* pc_ )
{
    auto& pc = *reinterpret_cast<PointCloud*>( pc_ );

    return reinterpret_cast<MRVector3f*>( pc.points.data() );
}

size_t mrPointCloudPointsNum( const MRPointCloud* pc_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    return pc.points.size();
}

const MRVector3f* mrPointCloudNormals( const MRPointCloud* pc_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    return reinterpret_cast<const MRVector3f*>( pc.normals.data() );
}

size_t mrPointCloudNormalsNum( const MRPointCloud* pc_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    return pc.normals.size();
}

const MRVertBitSet* mrPointCloudValidPoints( const MRPointCloud* pc_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );

    return reinterpret_cast<const MRVertBitSet*>( &pc.validPoints );
}

MRBox3f mrPointCloudComputeBoundingBox( const MRPointCloud* pc_, const MRAffineXf3f* toWorld_ )
{
    const auto& pc = *reinterpret_cast<const PointCloud*>( pc_ );
    const auto* toWorld = reinterpret_cast<const AffineXf3f*>( toWorld_ );

    const auto res = pc.computeBoundingBox( toWorld );
    return reinterpret_cast<const MRBox3f&>( res );
}

MRVertId mrPointCloudAddPoint( MRPointCloud* pc_, const MRVector3f* point_ )
{
    auto& pc = *reinterpret_cast<PointCloud*>( pc_ );
    const auto& point = *reinterpret_cast<const Vector3f*>( point_ );

    const auto res = pc.addPoint( point );
    return reinterpret_cast<const MRVertId&>( res );
}

void mrPointCloudFree( MRPointCloud* pc )
{
    delete reinterpret_cast<PointCloud*>( pc );
}
