#include "MRAncillaryLines.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRPolyline.h"

namespace MR
{

void AncillaryLines::make( Object &parent )
{
    reset();
    obj = std::make_shared<ObjectLines>();
    obj->setAncillary( true );
    obj->setFrontColor( SceneColors::get( SceneColors::Type::Labels ), false );
    obj->setPickable( false );
    parent.addChild( obj );
}

void AncillaryLines::make( Object &parent, const Contour3f& contour )
{
    make( parent );
    obj->setPolyline( std::make_shared<Polyline3>( contour ) );
}

void AncillaryLines::make( Object &parent, const Contours3f& contours )
{
    make( parent );
    obj->setPolyline( std::make_shared<Polyline3>( contours ) );
}

void AncillaryLines::reset()
{
    if ( obj )
        obj->detachFromParent();
    obj.reset();
}

void AncillaryLines::setContours( const Contours3f& contours )
{
    obj->setPolyline( std::make_shared<Polyline3>( contours ) );
}

void AncillaryLines::resetContours()
{
    obj->setPolyline( nullptr );
}

void AncillaryLines::setDepthTest( bool depthTest )
{
    obj->setVisualizeProperty( depthTest, VisualizeMaskType::DepthTest, ViewportMask::all() );
}

} // namespace MR
