#include "MRAncillaryLabels.h"
#include "MRSymbolMesh/MRObjectLabel.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRViewer.h"
#include "MRViewport.h"
#include "ImGuiMenu.h"
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

void AncillaryImGuiLabel::make( Object &parent, const PositionedText& text )
{
    make( text );
    parentXfConnection_ = parent.worldXfChangedSignal.connect( [&] ()
    {
        labelData_.position = parent.worldXf()( localPos_ );
    } );
    labelData_.position = parent.worldXf()( localPos_ );
}

void AncillaryImGuiLabel::make( std::shared_ptr<Object> parent, const PositionedText& text )
{
    make( text );
    parent_ = parent;
    if ( !parent )
        return;
    parentXfConnection_ = parent->worldXfChangedSignal.connect( [&, p = parent.get()] ()
    {
        labelData_.position = p->worldXf()( localPos_ );
    } );
    labelData_.position = parent->worldXf()( localPos_ );
}

void AncillaryImGuiLabel::make( const PositionedText& text )
{
    reset();
    labelData_ = text;
    localPos_ = text.position;
    // these parameters emulate plugins
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}

void AncillaryImGuiLabel::reset()
{
    disconnect();
    parent_.reset();
    if ( parentXfConnection_.connected() )
        parentXfConnection_.disconnect();
    labelData_ = {};
}

void AncillaryImGuiLabel::overrideParams( const ImGuiMeasurementIndicators::Params& params )
{
    overrideParams_ = params;
}

void AncillaryImGuiLabel::resetOverrideParams()
{
    overrideParams_.reset();
}

void AncillaryImGuiLabel::preDraw_()
{
    if ( labelData_.text.empty() )
        return;
    std::shared_ptr<Object> parent = parent_.lock();
    ViewportMask viewports = parent ? parent->globalVisibilityMask() : ViewportMask::all();
    if ( viewports == ViewportMask() )
        return;
    auto menu = getViewerInstance().getMenuPlugin();
    if ( !menu )
        return;

    ImGuiMeasurementIndicators::Params params;
    if ( overrideParams_ )
        params = *overrideParams_;
    if ( !params.list )
        return;
    if ( !overrideParams_ )
        params.colorTextOutline.a = 220;
    auto scaling = menu->menu_scaling();
    const ImGuiMeasurementIndicators::Text sWithI( labelData_.text );

    for ( const auto& vp : getViewerInstance().viewport_list )
    {
        if ( !viewports.contains( vp.id ) )
            continue;

        auto rect = vp.getViewportRect();
        ImVec2 minRect( rect.min.x,
                 getViewerInstance().framebufferSize.y - ( rect.min.y + height( rect ) ) );
        ImVec2 maxRect( rect.min.x + width( rect ),
                 getViewerInstance().framebufferSize.y - rect.min.y );

        Vector3f coord = vp.projectToViewportSpace( labelData_.position );
        auto viewerCoord = getViewerInstance().viewportToScreen( coord, vp.id );

        params.list->PushClipRect( minRect, maxRect );
        ImGuiMeasurementIndicators::text( ImGuiMeasurementIndicators::Element::both, scaling, params,
            ImVec2( viewerCoord.x, viewerCoord.y ), sWithI, {}, pivot_ );
        params.list->PopClipRect();
    }
}

} //namespace MR
