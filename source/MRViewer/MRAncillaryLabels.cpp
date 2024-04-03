#include "MRAncillaryLabels.h"
#include "MRMesh/MRObjectLabel.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRViewer.h"
#include "MRViewport.h"
#include "ImGuiMenu.h"
#include "MRImGuiMeasurementIndicators.h"
#include "imgui.h"

namespace MR
{

void AncillaryLabel::make( Object& parent, const PositionedText& text, bool depthTest )
{
    reset();
    obj = makeDetached( text, depthTest );
    parent.addChild( obj );
}

std::shared_ptr<ObjectLabel> AncillaryLabel::makeDetached( const PositionedText& text, bool depthTest )
{
    std::shared_ptr<ObjectLabel> obj = std::make_shared<ObjectLabel>();
    obj->setAncillary( true );
    obj->setFrontColor( SceneColors::get( SceneColors::Type::Labels ), false );
    obj->setLabel( text );
    obj->setPickable( false );
    obj->setVisualizeProperty( depthTest, VisualizeMaskType::DepthTest, ViewportMask::all() );
    return obj;
}

void AncillaryLabel::reset()
{
    if ( obj )
        obj->detachFromParent();
    obj.reset();
}

void AncillaryLabel::setText( const PositionedText& text )
{
    obj->setLabel( text );
}

void AncillaryLabel::resetText()
{
    obj->setLabel( {} );
}

void AncillaryLabel::setDepthTest( bool depthTest )
{
    obj->setVisualizeProperty( depthTest, VisualizeMaskType::DepthTest, ViewportMask::all() );
}

void AncillaryLabel::setPosition( const Vector3f& pos )
{
    obj->setLabel( PositionedText{ obj->getLabel().text, pos } );
}

void AncillaryImGuiLabel::make( Object& parent, const PositionedText& text )
{
    reset();
    make( text );
    parentXfConnection_ = parent.worldXfChangedSignal.connect( [&] ()
    {
        labelData_.position = parent.worldXf()( labelData_.position );
    } );
    labelData_.position = parent.worldXf()( labelData_.position );
}

void AncillaryImGuiLabel::make( const PositionedText& text )
{
    reset();
    labelData_ = text;
    // these parameters emulate plugins
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}

void AncillaryImGuiLabel::reset()
{
    disconnect();
    if ( parentXfConnection_.connected() )
        parentXfConnection_.disconnect();
    labelData_ = {};
}

void AncillaryImGuiLabel::preDraw_()
{
    if ( labelData_.text.empty() )
        return;
    auto menu = getViewerInstance().getMenuPlugin();
    if ( !menu )
        return;
    
    ImGuiMeasurementIndicators::Params params;
    if ( !params.list )
        return;
    params.colorTextOutline.a = 220;
    auto scaling = menu->menu_scaling();
    const ImGuiMeasurementIndicators::StringWithIcon sWithI( labelData_.text );

    for ( const auto& vp : getViewerInstance().viewport_list )
    {
        auto rect = vp.getViewportRect();
        ImVec2 minRect( rect.min.x,
                 getViewerInstance().framebufferSize.y - ( rect.min.y + height( rect ) ) );
        ImVec2 maxRect( rect.min.x + width( rect ),
                 getViewerInstance().framebufferSize.y - rect.min.y );


        Vector3f coord = vp.projectToViewportSpace( labelData_.position );
        auto viewerCoord = getViewerInstance().viewportToScreen( coord, vp.id );

        params.list->PushClipRect( minRect, maxRect );
        ImGuiMeasurementIndicators::text( ImGuiMeasurementIndicators::Element::both, scaling, params, ImVec2( viewerCoord.x, viewerCoord.y ), sWithI );
        params.list->PopClipRect();
    }
}

} //namespace MR
