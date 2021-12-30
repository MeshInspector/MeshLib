#include "MRDistanceMapParams.h"

namespace MR
{

MeshToDistanceMapParams::MeshToDistanceMapParams( const Vector3f& direction, const Vector2i& res, const MeshPart& mp, bool usePreciseBoundingBox )
{
    auto pair = direction.perpendicular();
    Matrix3f rot{ pair.first,pair.second,direction };
    auto orgSize = orgSizeFromMeshPart_( rot, mp, usePreciseBoundingBox );
    initFromSize_( { rot,orgSize.first }, res, orgSize.second );
}

MeshToDistanceMapParams::MeshToDistanceMapParams( const Matrix3f& rotation, const Vector3f& origin, const Vector2i& resolution, const Vector2f& size )
{
    initFromSize_( { rotation,origin }, resolution, size );
}

MeshToDistanceMapParams::MeshToDistanceMapParams( const Matrix3f& rotation, const Vector3f& origin, const Vector2f& pixelSize, const Vector2i& resolution )
{
    initFromSize_( { rotation,origin }, resolution, { pixelSize.x * resolution.x,pixelSize.y * resolution.y } );
}

MeshToDistanceMapParams::MeshToDistanceMapParams( const Matrix3f& rotation, const Vector2i& resolution, const MeshPart& mp, bool usePreciseBoundingBox )
{
    // input matrix should be orthonormal!
    assert( std::fabs( std::fabs( rotation.det() ) - 1.f ) < 1e-5 );

    auto orgSize = orgSizeFromMeshPart_( rotation, mp, usePreciseBoundingBox );
    initFromSize_( { rotation,orgSize.first }, resolution, orgSize.second );
}

MeshToDistanceMapParams::MeshToDistanceMapParams( const AffineXf3f& xf, const Vector2i& resolution, const Vector2f& size )
{
    initFromSize_( xf, resolution, size );
}

MeshToDistanceMapParams::MeshToDistanceMapParams( const AffineXf3f& xf, const Vector2f& pixelSize, const Vector2i& resolution )
{
    initFromSize_( xf, resolution, { pixelSize.x * resolution.x,pixelSize.y * resolution.y } );
}

std::pair<Vector3f, Vector2f> MeshToDistanceMapParams::orgSizeFromMeshPart_( const Matrix3f& rotation, const MeshPart& mp, bool presiceBox ) const
{
    auto orientation = AffineXf3f( rotation, Vector3f() );
    Box3f box;
    if ( presiceBox )
    {
        box = mp.mesh.computeBoundingBox( mp.region, &orientation );
    }
    else
    {
        box = transformed( mp.mesh.getBoundingBox(), orientation );
    }
    return { orientation.inverse()( box.min ),{box.max.x - box.min.x,box.max.y - box.min.y} };
}

void MeshToDistanceMapParams::initFromSize_( const AffineXf3f& worldOrientation, const Vector2i& res, const Vector2f& size )
{
    resolution = res;
    orgPoint = worldOrientation.b;
    direction = worldOrientation.A.z;
    xRange = worldOrientation.A.x * size.x;
    yRange = worldOrientation.A.y * size.y;
}


ContourToDistanceMapParams::ContourToDistanceMapParams( const Vector2i& dmapSize_,
    const Vector2f& oriPoint_, const Vector2f& areaSize, bool withSign_ /*= false */ ) :
    pixelSize{ areaSize.x / dmapSize_.x, areaSize.y / dmapSize_.y },
    resolution{ dmapSize_ },
    orgPoint{ oriPoint_ },
    withSign{ withSign_ }
{
    assert( resolution.x > 0 && resolution.y > 0 );
}

ContourToDistanceMapParams::ContourToDistanceMapParams( const Vector2i& dmapSize_,
    const Box2f& box, bool withSign_ /*= false */ ) :
    pixelSize{ ( box.max.x - box.min.x ) / dmapSize_.x, ( box.max.y - box.min.y ) / dmapSize_.y },
    resolution{ dmapSize_ },
    orgPoint{ box.min },
    withSign{ withSign_ }
{
    assert( box.valid() );
    assert( resolution.x > 0 && resolution.y > 0 );
}

ContourToDistanceMapParams::ContourToDistanceMapParams( const Vector2i& dmapSize_,
    const Contours2f& contours, float offset, bool withSign_ /*= false */ ) :
    pixelSize{},
    resolution{ dmapSize_ },
    orgPoint{},
    withSign{ withSign_ }
{
    Box2f box;
    for ( const auto& c : contours )
        for ( const auto& p : c )
            box.include( p );

    box.min -= Vector2f::diagonal( offset );
    box.max += Vector2f::diagonal( offset );

    orgPoint = box.min;

    pixelSize.x = ( box.max.x - box.min.x ) / resolution.x;
    pixelSize.y = ( box.max.y - box.min.y ) / resolution.y;
}

ContourToDistanceMapParams::ContourToDistanceMapParams( float pixelSize,
    const Contours2f& contours, float offset, bool withSign_ /*= false */ ) :
    pixelSize{pixelSize,pixelSize},
    resolution{},
    orgPoint{},
    withSign{ withSign_ }
{
    Box2f box;
    for ( const auto& c : contours )
        for ( const auto& p : c )
            box.include( p );

    box.min -= Vector2f::diagonal( offset );
    box.max += Vector2f::diagonal( offset );

    orgPoint = box.min;

    resolution.x = int( ( box.max.x - box.min.x ) / pixelSize );
    resolution.y = int( ( box.max.y - box.min.y ) / pixelSize );
}

ContourToDistanceMapParams::ContourToDistanceMapParams( const DistanceMapToWorld& toWorld ) :
    pixelSize{ toWorld.pixelXVec.x, toWorld.pixelYVec.y },
    resolution{},
    orgPoint{ toWorld.orgPoint.x, toWorld.orgPoint.y },
    withSign{}
{
}

DistanceMapToWorld::DistanceMapToWorld( const MeshToDistanceMapParams& params ) :
    orgPoint{ params.orgPoint },
    pixelXVec{ params.xRange / float( params.resolution.x ) },
    pixelYVec{ params.yRange / float( params.resolution.y ) },
    direction{ params.direction }
{
    assert( params.resolution.x > 0 && params.resolution.y > 0 );
}

DistanceMapToWorld::DistanceMapToWorld( const ContourToDistanceMapParams& params ) :
    orgPoint{ Vector3f{ params.orgPoint.x, params.orgPoint.y, 0.F } },
    pixelXVec{ params.pixelSize.x , 0.F, 0.F },
    pixelYVec{ 0.F, params.pixelSize.y, 0.F },
    direction{ 0.F, 0.F, 1.F }
{
    assert( params.resolution.x > 0 && params.resolution.y > 0 );
}

}
