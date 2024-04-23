#include "MRMoveObjectByMouseImpl.h"
#include "MRViewer/MRMouse.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRIntersection.h"
#include "MRMesh/MRVisualObject.h"
#include "MRPch/MRSpdlog.h"

namespace
{
// translation multiplier that limits its maximum value depending on object size
// the constant duplicates value defined in ImGuiMenu implementation
constexpr float cMaxTranslationMultiplier = 0xC00;
}

namespace MR
{

void MoveObjectByMouseImpl::onDrawDialog( float /* menuScaling */ )
{
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
}

bool MoveObjectByMouseImpl::onMouseDown( MouseButton button, int modifier )
{
    if ( button != Viewer::MouseButton::Left )
        return false;

    viewer->select_hovered_viewport();

    auto [obj, pick] = viewer->viewport().pick_render_object();

    if ( !obj || !onPick( obj, pick ) || !obj )
        return false;

    visualizeVectors_.clear();
    angle_ = 0.f;
    shift_ = 0.f;

    obj_ = obj;
    objWorldXf_ = obj_->worldXf();
    worldStartPoint_ = objWorldXf_( pick.point );
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

bool MoveObjectByMouseImpl::onMouseMove( int x, int y )
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
        AffineXf3f worldXfA = AffineXf3f::linear( objWorldXf_.A );
        AffineXf3f toBboxCenter = AffineXf3f::translation( worldXfA( bboxCenter_ ) );
        obj_->setWorldXf( AffineXf3f::translation( objWorldXf_.b ) * toBboxCenter * rotation * toBboxCenter.inverse() * worldXfA );
    }
    else
    {
        shift_ = ( worldEndPoint - worldStartPoint_ ).length();
        setVisualizeVectors_( { worldStartPoint_, worldEndPoint } );

        auto worldXf = AffineXf3f::translation( worldEndPoint - worldStartPoint_ ) * objWorldXf_;

        // Clamp movement.
        float minSizeDim = 0;
        if ( auto worldBox = transformed( obj_->getBoundingBox(), worldXf ); worldBox.valid() ) // Feature objects give an invalid box.
        {
            auto wbsize = worldBox.size();
            minSizeDim = wbsize.length();
        }

        if ( minSizeDim == 0 )
            minSizeDim = 1.f;

        for ( auto i = 0; i < 3; i++ )
            worldXf.b[i] = std::clamp( worldXf.b[i], -cMaxTranslationMultiplier * minSizeDim, +cMaxTranslationMultiplier * minSizeDim );

        obj_->setWorldXf( worldXf );
    }

    return true;
}

bool MoveObjectByMouseImpl::onMouseUp( MouseButton btn, int /*modifiers*/ )
{
    if ( !obj_ || btn != Viewer::MouseButton::Left )
        return false;

    auto newWorldXf = obj_->worldXf();
    obj_->setWorldXf( objWorldXf_ );
    AppendHistory<ChangeXfAction>( "Change Xf", obj_ );
    obj_->setWorldXf( newWorldXf );

    obj_ = nullptr;
    transformMode_ = TransformMode::None;

    return true;
}

void MoveObjectByMouseImpl::cancel()
{
    if ( !obj_ )
        return;
    obj_->setWorldXf( objWorldXf_ );
    obj_ = nullptr;
    transformMode_ = TransformMode::None;
}

bool MoveObjectByMouseImpl::onPick( std::shared_ptr<VisualObject>& obj, PointOnObject& )
{
    return !obj->isAncillary();
}

void MoveObjectByMouseImpl::setVisualizeVectors_( std::vector<Vector3f> worldPoints )
{
    visualizeVectors_.clear();
    for ( const auto& p : worldPoints )
    {
        const Vector3f screenPoint = viewer->viewportToScreen( viewer->viewport().projectToViewportSpace( p ), viewer->viewport().id );
        visualizeVectors_.push_back( ImVec2( screenPoint.x, screenPoint.y ) );
    }
}

}
