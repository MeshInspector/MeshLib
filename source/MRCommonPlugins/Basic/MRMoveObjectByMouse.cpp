#include "MRMoveObjectByMouse.h"
#include "MRViewer/MRMouse.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRMesh/MRSceneColors.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRIntersection.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/ImGuiHelpers.h"

namespace
{
// translation multiplier that limits its maximum value depending on object size
// the constant duplicates value defined in ImGuiMenu implementation
constexpr float cMaxTranslationMultiplier = 0xC00;
}

namespace MR
{

MoveObjectByMouse::MoveObjectByMouse() :
    PluginParent( "Move object", StatePluginTabs::Basic )
{
}

void MoveObjectByMouse::drawDialog( float menuScaling, ImGuiContext*)
{
    auto menuWidth = 400.f * menuScaling;
    if ( !ImGui::BeginCustomStatePlugin( plugin_name.c_str(), &dialogIsOpen_, { .collapsed = &dialogIsCollapsed_, .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    ImGui::Text( "%s", "Click and hold LMB on object to move" );
    ImGui::Text( "%s", "Click CTRL + LMB and hold LMB on object to rotate" );

    if ( transformMode_ != TransformMode::None )
    {
        auto drawList = ImGui::GetBackgroundDrawList();
        drawList->AddPolyline( visualizeVectors_.data(), int( visualizeVectors_.size() ),
                               SceneColors::get( SceneColors::Labels ).getUInt32(), ImDrawFlags_None, 1.f );
    }
    if ( transformMode_ == TransformMode::Translation )
        ImGui::SetTooltip( "Distance : %.3f", shift_ );
    if ( transformMode_ == TransformMode::Rotation )
        ImGui::SetTooltip( "Angle : %.3f", angle_ );

    ImGui::EndCustomStatePlugin();
}

bool MoveObjectByMouse::onMouseDown_( MouseButton button, int modifier )
{
    if ( button != Viewer::MouseButton::Left )
        return false;

    const auto [obj, pick] = viewer->viewport().pick_render_object();

    if ( !obj || obj->isAncillary() )
        return false;

    visualizeVectors_.clear();
    angle_ = 0.f;
    shift_ = 0.f;

    obj_ = obj;
    objXf_ = obj_->xf();
    worldStartPoint_ = obj_->worldXf()( pick.point );
    viewportStartPointZ_ = viewer->viewport().projectToViewportSpace( worldStartPoint_ ).z;
    
    transformMode_ = ( modifier == GLFW_MOD_CONTROL ) ? TransformMode::Rotation : TransformMode::Translation;
    
    if ( transformMode_ == TransformMode::Rotation )
    {
        bboxCenter_ = obj_->getBoundingBox().center();
        worldBboxCenter_ = obj_->worldXf()( bboxCenter_ );
        auto viewportBboxCenter = viewer->viewport().projectToViewportSpace( worldBboxCenter_ );

        auto bboxCenterAxis = viewer->viewport().unprojectPixelRay( Vector2f( viewportBboxCenter.x, viewportBboxCenter.y ) );
        rotationPlane_ = Plane3f::fromDirAndPt( bboxCenterAxis.d.normalized(), worldBboxCenter_ );

        auto viewportStartPoint = viewer->viewport().projectToViewportSpace( worldStartPoint_ );
        auto startAxis = viewer->viewport().unprojectPixelRay( Vector2f( viewportStartPoint.x, viewportStartPoint.y ) );

        if ( auto crossPL = intersection( rotationPlane_, startAxis ) )
            worldStartPoint_ = *crossPL;
        else
            spdlog::warn( "Bad cross start axis and rotation plane" );

        setVisualizeVectors_( { worldBboxCenter_, worldStartPoint_, worldBboxCenter_, worldStartPoint_ } );
    }
    else
        setVisualizeVectors_( { worldStartPoint_, worldStartPoint_ } );

    return true;
}

bool MoveObjectByMouse::onMouseMove_( int x, int y )
{
    if ( !obj_ )
        return false;
    
    auto viewportEnd = viewer->screenToViewport( Vector3f( float( x ), float( y ), 0.f ), viewer->viewport().id );
    auto worldEndPoint = viewer->viewport().unprojectFromViewportSpace( { viewportEnd.x, viewportEnd.y, viewportStartPointZ_ } );

    if ( transformMode_ == TransformMode::Rotation )
    {
        auto endAxis = viewer->viewport().unprojectPixelRay( Vector2f( viewportEnd.x, viewportEnd.y ) );
        if ( auto crossPL = intersection( rotationPlane_, endAxis ) )
            worldEndPoint = *crossPL;
        else
            spdlog::warn( "Bad cross end axis and rotation plane" );

        const Vector3f vectorStart = worldStartPoint_ - worldBboxCenter_;
        const Vector3f vectorEnd = worldEndPoint - worldBboxCenter_;
        const float abSquare = vectorStart.length() * vectorEnd.length();
        if ( abSquare < 1.e-6 )
            angle_ = 0.f;
        else
            angle_ = angle( vectorStart, vectorEnd );
        
        if ( dot( rotationPlane_.n, cross( vectorStart, vectorEnd ) ) > 0.f )
            angle_ = 2.f * PI_F - angle_;
        
        angle_ = angle_ / PI_F * 180.f;

        setVisualizeVectors_( { worldBboxCenter_, worldStartPoint_, worldBboxCenter_, worldEndPoint } );

        AffineXf3f rotation = AffineXf3f::linear( Matrix3f::rotation( vectorStart, worldEndPoint - worldBboxCenter_ ) );
        AffineXf3f xfA = AffineXf3f::linear( objXf_.A );
        AffineXf3f toBboxCenter = AffineXf3f::translation( xfA( bboxCenter_ ) );
        obj_->setXf( AffineXf3f::translation(objXf_.b) * toBboxCenter * rotation * toBboxCenter.inverse() * xfA );
    }
    else
    {
        shift_ = ( worldEndPoint - worldStartPoint_ ).length();
        setVisualizeVectors_( { worldStartPoint_, worldEndPoint } );

        auto xf = AffineXf3f::translation( worldEndPoint - worldStartPoint_ ) * objXf_;
        auto worldXf = obj_->parent() ? obj_->parent()->worldXf() * xf : xf;

        auto wbsize = transformed( obj_->getBoundingBox(), worldXf ).size();
        auto minSizeDim = wbsize.length();
        if ( minSizeDim == 0 )
            minSizeDim = 1.f;

        for ( auto i = 0; i < 3; i++ )
            xf.b[i] = std::clamp( xf.b[i], -cMaxTranslationMultiplier * minSizeDim, +cMaxTranslationMultiplier * minSizeDim );

        obj_->setXf( xf );
    }

    return true;
}

bool MoveObjectByMouse::onMouseUp_( MouseButton btn, int /*modifiers*/ )
{
    if ( !obj_ || btn != Viewer::MouseButton::Left )
        return false;

    auto newXf = obj_->xf();
    obj_->setXf( objXf_ );
    AppendHistory<ChangeXfAction>( "Change Xf", obj_ );
    obj_->setXf( newXf );

    obj_ = nullptr;
    transformMode_ = TransformMode::None;

    return true;
}

void MoveObjectByMouse::setVisualizeVectors_( std::vector<Vector3f> worldPoints )
{
    visualizeVectors_.clear();
    for ( const auto& p : worldPoints )
    {
        const Vector3f screenPoint = viewer->viewportToScreen( viewer->viewport().projectToViewportSpace( p ), viewer->viewport().id );
        visualizeVectors_.push_back( ImVec2( screenPoint.x, screenPoint.y ) );
    }
}

MR_REGISTER_RIBBON_ITEM( MoveObjectByMouse )

}
