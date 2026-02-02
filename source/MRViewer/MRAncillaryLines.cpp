#include "MRAncillaryLines.h"
#include "MRColorTheme.h"

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

void AncillaryLines::colorizeAxes()
{
    if ( !obj || !obj->polyline() )
        return;

    const auto& polyline = *obj->polyline();
    const auto ueCount = polyline.topology.lastNotLoneUndirectedEdge() + 1;
    UndirectedEdgeColors colorMap( ueCount, Color::black() );
    for ( auto ue = 0_ue; ue < ueCount; ++ue )
    {
        const auto vec = polyline.edgeVector( ue );
        if ( vec.y == 0.f && vec.z == 0.f )
            colorMap[ue] = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisX );
        else if ( vec.x == 0.f && vec.z == 0.f )
            colorMap[ue] = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisY );
        else if ( vec.x == 0.f && vec.y == 0.f )
            colorMap[ue] = ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::AxisZ );
    }
    obj->setLinesColorMap( std::move( colorMap ) );
    obj->setColoringType( ColoringType::LinesColorMap );
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
