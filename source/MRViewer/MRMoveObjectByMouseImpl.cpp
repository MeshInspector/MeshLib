#include "MRMoveObjectByMouseImpl.h"
#include "MRViewer/MRMouse.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRViewer/MRMouseController.h"
#include "MRViewer/MRViewerInstance.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRIntersection.h"
#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MR2to3.h"
#include "MRPch/MRSpdlog.h"

namespace
{
// translation multiplier that limits its maximum value depending on object size
// the constant duplicates value defined in ImGuiMenu implementation
constexpr float cMaxTranslationMultiplier = 0xC00;

// special value for screenStartPoint_
constexpr MR::Vector2i cNoPoint{ std::numeric_limits<int>::max(), 0 };

}

namespace MR
{

void MoveObjectByMouseImpl::onDrawDialog( float /* menuScaling */ ) const
{
    if ( !isMoving() )
        return;
    if ( transformMode_ != TransformMode::None )
    {
        auto drawList = ImGui::GetBackgroundDrawList();
        drawList->AddPolyline( visualizeVectors_.data(), int( visualizeVectors_.size() ),
                               SceneColors::get( SceneColors::Labels ).getUInt32(), ImDrawFlags_None, 1.f );
    }
    if ( transformMode_ == TransformMode::Translation )
        ImGui::SetTooltip( "Distance : %s", valueToString<LengthUnit>( shift_ ).c_str() );
    if ( transformMode_ == TransformMode::Rotation )
        ImGui::SetTooltip( "Angle : %s", valueToString<AngleUnit>( angle_ ).c_str() );
}

bool MoveObjectByMouseImpl::onMouseDown( MouseButton button, int modifiers )
{
    if ( button != Viewer::MouseButton::Left )
        return false;

    Viewer& viewer = getViewerInstance();
    Viewport& viewport = viewer.viewport();
    
    screenStartPoint_ = minDistance() > 0 ? viewer.mouseController().getMousePos() : cNoPoint;

    auto [obj, pick] = viewport.pick_render_object();

    if ( !obj || !onPick_( obj, pick, modifiers ) || !obj )
        return false;

    objects_ = getObjects_( obj, pick, modifiers );

    visualizeVectors_.clear();
    angle_ = 0.f;
    shift_ = 0.f;

    obj_ = obj;
    newWorldXf_ = objWorldXf_ = obj_->worldXf();
    worldStartPoint_ = objWorldXf_( pick.point );
    viewportStartPointZ_ = viewport.projectToViewportSpace( worldStartPoint_ ).z;
    objectsXfs_.clear();
    for ( std::shared_ptr<Object>& f : objects_ )
        objectsXfs_.push_back( f->worldXf() );

    transformMode_ = ( modifiers & GLFW_MOD_CONTROL ) != 0 ? TransformMode::Rotation : TransformMode::Translation;

    if ( transformMode_ == TransformMode::Rotation )
    {
        bboxCenter_ = obj_->getBoundingBox().center();
        worldBboxCenter_ = obj_->worldXf()( bboxCenter_ );
        auto viewportBboxCenter = viewport.projectToViewportSpace( worldBboxCenter_ );

        auto bboxCenterAxis = viewport.unprojectPixelRay( Vector2f( viewportBboxCenter.x, viewportBboxCenter.y ) );
        rotationPlane_ = Plane3f::fromDirAndPt( bboxCenterAxis.d.normalized(), worldBboxCenter_ );

        auto viewportStartPoint = viewport.projectToViewportSpace( worldStartPoint_ );
        auto startAxis = viewport.unprojectPixelRay( Vector2f( viewportStartPoint.x, viewportStartPoint.y ) );

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

    Viewer& viewer = getViewerInstance();
    Viewport& viewport = viewer.viewport();

    if ( screenStartPoint_ != cNoPoint &&
         ( screenStartPoint_ - viewer.mouseController().getMousePos() ).lengthSq() <
             minDistance() * minDistance() )
        return true;
    screenStartPoint_ = cNoPoint;

    auto viewportEnd = viewer.screenToViewport( Vector3f( float( x ), float( y ), 0.f ), viewport.id );
    auto worldEndPoint = viewport.unprojectFromViewportSpace( { viewportEnd.x, viewportEnd.y, viewportStartPointZ_ } );
    AffineXf3f worldXf;

    if ( transformMode_ == TransformMode::Rotation )
    {
        auto endAxis = viewport.unprojectPixelRay( Vector2f( viewportEnd.x, viewportEnd.y ) );
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

        setVisualizeVectors_( { worldBboxCenter_, worldStartPoint_, worldBboxCenter_, worldEndPoint } );

        AffineXf3f rotation = AffineXf3f::linear( Matrix3f::rotation( vectorStart, worldEndPoint - worldBboxCenter_ ) );
        AffineXf3f worldXfA = AffineXf3f::linear( objWorldXf_.A );
        AffineXf3f toBboxCenter = AffineXf3f::translation( worldXfA( bboxCenter_ ) );
        worldXf = AffineXf3f::translation( objWorldXf_.b ) * toBboxCenter * rotation * toBboxCenter.inverse() * worldXfA;
    }
    else
    {
        shift_ = ( worldEndPoint - worldStartPoint_ ).length();
        setVisualizeVectors_( { worldStartPoint_, worldEndPoint } );

        worldXf = AffineXf3f::translation( worldEndPoint - worldStartPoint_ ) * objWorldXf_;

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
    }

    newWorldXf_ = worldXf;
    setWorldXf_( worldXf, false );

    return true;
}

bool MoveObjectByMouseImpl::onMouseUp( MouseButton button, int /*modifiers*/ )
{
    if ( !obj_ || button != Viewer::MouseButton::Left )
        return false;

    if ( screenStartPoint_ != cNoPoint )
    {
        clear_();
        return false;
    }

    resetWorldXf_();
    setWorldXf_( newWorldXf_, true );

    clear_();

    return true;
}

bool MoveObjectByMouseImpl::isMoving() const
{
    return screenStartPoint_ == cNoPoint;
}

void MoveObjectByMouseImpl::cancel()
{
    if ( !obj_ )
        return;
    resetWorldXf_();
    clear_();
}

bool MoveObjectByMouseImpl::onPick_( std::shared_ptr<VisualObject>& obj, PointOnObject&, int )
{
    return !obj->isAncillary();
}

std::vector<std::shared_ptr<Object>> MoveObjectByMouseImpl::getObjects_( 
    const std::shared_ptr<VisualObject>& obj, const PointOnObject &, int )
{
    return { obj };
}

void MoveObjectByMouseImpl::clear_()
{
    obj_ = nullptr;
    transformMode_ = TransformMode::None;
    objects_.clear();
    objectsXfs_.clear();
    screenStartPoint_ = {};
}

void MoveObjectByMouseImpl::setWorldXf_( AffineXf3f worldXf, bool history )
{
    std::unique_ptr<ScopeHistory> scope = history ? std::make_unique<ScopeHistory>( "Change Xf" ) : nullptr;
    auto itXf = objectsXfs_.begin();
    for ( std::shared_ptr<Object>& obj : objects_ )
    {
        if ( history )
            AppendHistory<ChangeXfAction>( "Change Xf", obj );
        if ( obj == obj_ )
            obj->setWorldXf( worldXf );
        else
            obj->setWorldXf( worldXf * objWorldXf_.inverse() * *itXf );
        itXf++;
    }
}

void MoveObjectByMouseImpl::resetWorldXf_()
{
    auto itXf = objectsXfs_.begin();
    for ( std::shared_ptr<Object>& f : objects_ )
        f->setWorldXf( *itXf++ );
}

void MoveObjectByMouseImpl::setVisualizeVectors_( std::vector<Vector3f> worldPoints )
{
    Viewer& viewer = getViewerInstance();
    Viewport& viewport = viewer.viewport();
    visualizeVectors_.clear();
    for ( const auto& p : worldPoints )
    {
        const Vector3f screenPoint = viewer.viewportToScreen(
            viewport.projectToViewportSpace( p ), viewport.id );
        visualizeVectors_.push_back( ImVec2( screenPoint.x, screenPoint.y ) );
    }
}

}
