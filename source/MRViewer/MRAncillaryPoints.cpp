#include "MRAncillaryPoints.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRPointCloud.h"

namespace MR
{

void AncillaryPoints::make( Object& parent )
{
    reset();
    obj = std::make_shared<ObjectPoints>();
    obj->setPointCloud( std::make_shared<PointCloud>() );
    obj->setAncillary( true );
    obj->setFrontColor( SceneColors::get( SceneColors::Type::Labels ), false );
    obj->setPickable( false );
    parent.addChild( obj );
}

void AncillaryPoints::reset()
{
    if ( obj )
        obj->detachFromParent();
    obj.reset();
}

void AncillaryPoints::addPoint( const Vector3f& point )
{
    obj->varPointCloud()->addPoint( point );
    obj->setDirtyFlags( DIRTY_POSITION );
}

void AncillaryPoints::addPoint( const Vector3f& point, const Color& color )
{
    auto colorMap = obj->getVertsColorMap();
    assert( colorMap.size() == obj->pointCloud()->points.size() );
    obj->varPointCloud()->addPoint( point );
    colorMap.push_back( color );
    obj->setVertsColorMap( colorMap );
    obj->setDirtyFlags( DIRTY_POSITION );
}

void AncillaryPoints::addPoints( const std::vector<Vector3f>& points )
{
    for ( const auto& point : points )
        obj->varPointCloud()->addPoint( point );
    obj->setDirtyFlags( DIRTY_POSITION );
}

void AncillaryPoints::addPoints( const std::vector<Vector3f>& points, const std::vector<Vector4f>& colors )
{
    assert( points.size() == colors.size() );

    auto colorMap = obj->getVertsColorMap();
    assert( colorMap.size() == obj->pointCloud()->points.size() );
    colorMap.reserve( colorMap.size() + points.size() );
    auto& oldPoints = obj->varPointCloud()->points;
    oldPoints.reserve( oldPoints.size() + points.size() );

    for ( size_t i = 0; i < points.size(); ++i )
    {
        obj->varPointCloud()->addPoint( points[i] );
        colorMap.push_back( Color( colors[i][0], colors[i][1], colors[i][2], colors[i][3] ) );
    }

    obj->setVertsColorMap( colorMap );
    obj->setDirtyFlags( DIRTY_POSITION );
}

void AncillaryPoints::setDepthTest( bool depthTest )
{
    obj->setVisualizeProperty( depthTest, VisualizeMaskType::DepthTest, ViewportMask::all() );
}

} //namespace MR
