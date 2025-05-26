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

namespace MR
{

void MoveObjectByMouseImpl::onDrawDialog( float menuScaling ) const
{
    if ( deadZonePixelRadius_ > 0.0f )
    {
        std::vector<std::shared_ptr<Object>> tempObjects;
        TransformMode expectedMode = transformMode_;

        if ( objects_.empty() )
        {
            int mods = 0;
            if ( ImGui::GetIO().KeyMods & ImGuiMod_Ctrl )
                mods |= GLFW_MOD_CONTROL;
            if ( ImGui::GetIO().KeyMods & ImGuiMod_Shift )
                mods |= GLFW_MOD_SHIFT;
            if ( ImGui::GetIO().KeyMods & ImGuiMod_Alt )
                mods |= GLFW_MOD_ALT;
            if ( ImGui::GetIO().KeyMods & ImGuiMod_Super )
                mods |= GLFW_MOD_SUPER;
            expectedMode = modeFromPickModifiers_( mods );
            if ( expectedMode == TransformMode::Rotation || expectedMode == TransformMode::UniformScale || expectedMode == TransformMode::NonUniformScale )
                pickObjects_( tempObjects, mods );
        }
        const auto& objs = objects_.empty() ? tempObjects : objects_;

        if ( !objs.empty() && ( expectedMode == TransformMode::Rotation || expectedMode == TransformMode::UniformScale || expectedMode == TransformMode::NonUniformScale ) )
        {
            ViewportId vpId = getViewerInstance().viewport().id;
            Vector3f centerPoint = xfCenterPoint_;
            if ( objects_.empty() )
            {
                setCenterPoint_( objs, centerPoint );
                vpId = getViewerInstance().getHoveredViewportId();
            }

            const auto& vp = getViewerInstance().viewport( vpId );
            auto screenPos = getViewerInstance().viewportToScreen( vp.projectToViewportSpace( centerPoint ), vpId );

            auto drawList = ImGui::GetBackgroundDrawList();
            drawList->AddCircleFilled( ImVec2( screenPos.x, screenPos.y ), menuScaling * deadZonePixelRadius_, Color::gray().scaledAlpha( 0.5f ).getUInt32() );
            if ( deadZonePixelRadius_ * 0.5f > 4.0f )
            {
                drawList->AddCircleFilled( ImVec2( screenPos.x, screenPos.y ), menuScaling * 4.0f, Color::red().getUInt32() );
            }
        }
    }

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
    if ( transformMode_ == TransformMode::UniformScale )
        ImGui::SetTooltip( "Uniform Scale : %s", valueToString<RatioUnit>( scale_ ).c_str() );
    if ( transformMode_ == TransformMode::NonUniformScale )
        ImGui::SetTooltip( "Non-Uniform Scale : %s", valueToString<RatioUnit>( scale_ ).c_str() );
}

bool MoveObjectByMouseImpl::onMouseDown( MouseButton button, int modifiers )
{
    Viewer& viewer = getViewerInstance();
    Viewport& viewport = viewer.viewport();

    cancel();
    transformMode_ = pick_( button, modifiers );
    if ( transformMode_ == TransformMode::None )
    {
        clear_();
        return false;
    }

    currentButton_ = button;
    screenStartPoint_ = viewer.mouseController().getMousePos();

    auto viewportStartPoint = viewport.projectToViewportSpace( worldStartPoint_ );
    viewportStartPointZ_ = viewportStartPoint.z;

    Vector3f viewportCenterPoint;
    if ( transformMode_ == TransformMode::Rotation || transformMode_ == TransformMode::UniformScale || transformMode_ == TransformMode::NonUniformScale )
    {
        viewportCenterPoint = viewport.projectToViewportSpace( xfCenterPoint_ );

        if ( deadZonePixelRadius_ > 0.0f )
        {
            float realDeadZone = deadZonePixelRadius_;
            if ( const auto& menu = viewer.getMenuPlugin() )
                realDeadZone *= menu->menu_scaling();
            if ( to2dim( viewportStartPoint - viewportCenterPoint ).lengthSq() <= sqr( realDeadZone ) )
            {
                clear_();
                return false;
            }
        }
    }

    angle_ = 0.f;
    shift_ = 0.f;
    scale_ = 1.f;
    currentXf_ = {};
    initialXfs_.clear();
    connections_.clear();
    xfChanged_ = false;
    for ( std::shared_ptr<Object>& obj : objects_ )
    {
        initialXfs_.push_back( obj->worldXf() );
        connections_.emplace_back( obj->worldXfChangedSignal.connect( [this]
        {
            if ( !changingXfFromMouseMove_ )
                clear_(); // stop mouse dragging if the transformation was changed from outside (e.g. undo)
        } ) );
    }

    if ( transformMode_ == TransformMode::Rotation || transformMode_ == TransformMode::UniformScale || transformMode_ == TransformMode::NonUniformScale )
    {
        Line3f centerAxis = viewport.unprojectPixelRay( Vector2f( viewportCenterPoint.x, viewportCenterPoint.y ) );
        referencePlane_ = Plane3f::fromDirAndPt( centerAxis.d.normalized(), xfCenterPoint_ );

        Line3f startAxis = viewport.unprojectPixelRay( Vector2f( viewportStartPoint.x, viewportStartPoint.y ) );

        if ( auto crossPL = intersection( referencePlane_, startAxis ) )
            worldStartPoint_ = *crossPL;
        else
            spdlog::warn( "Bad cross start axis and rotation plane" );

        setVisualizeVectors_( { xfCenterPoint_, worldStartPoint_, xfCenterPoint_, worldStartPoint_ } );
    }
    else // if ( transformMode_ == TransformMode::Translation )
        setVisualizeVectors_( { worldStartPoint_, worldStartPoint_ } );

    return true;
}

bool MoveObjectByMouseImpl::onMouseMove( int x, int y )
{
    if ( transformMode_ == TransformMode::None )
        return false;

    Viewer& viewer = getViewerInstance();
    Viewport& viewport = viewer.viewport();

    if ( !xfChanged_ &&
         ( screenStartPoint_ - viewer.mouseController().getMousePos() ).lengthSq() <
             minDistance() * minDistance() )
        return true; // mouse has moved less than threshold to change objects' transformation

    auto viewportEnd = viewer.screenToViewport( Vector3f( float( x ), float( y ), 0.f ), viewport.id );
    auto worldEndPoint = viewport.unprojectFromViewportSpace( { viewportEnd.x, viewportEnd.y, viewportStartPointZ_ } );

    if ( transformMode_ == TransformMode::Rotation )
    {
        auto endAxis = viewport.unprojectPixelRay( Vector2f( viewportEnd.x, viewportEnd.y ) );
        if ( auto crossPL = intersection( referencePlane_, endAxis ) )
            worldEndPoint = *crossPL;
        else
            spdlog::warn( "Bad cross end axis and rotation plane" );

        const Vector3f vectorStart = worldStartPoint_ - xfCenterPoint_;
        const Vector3f vectorEnd = worldEndPoint - xfCenterPoint_;
        const float abSquare = vectorStart.length() * vectorEnd.length();
        if ( abSquare < 1.e-6 )
            angle_ = 0.f;
        else
            angle_ = angle( vectorStart, vectorEnd );

        if ( dot( referencePlane_.n, cross( vectorStart, vectorEnd ) ) > 0.f )
            angle_ = 2.f * PI_F - angle_;

        setVisualizeVectors_( { xfCenterPoint_, worldStartPoint_, xfCenterPoint_, worldEndPoint } );

        // Rotate around center point (e.g. bounding box center)
        currentXf_ = AffineXf3f::xfAround( Matrix3f::rotation( vectorStart, worldEndPoint - xfCenterPoint_ ), xfCenterPoint_ );
    }
    else if ( transformMode_ == TransformMode::UniformScale || transformMode_ == TransformMode::NonUniformScale )
    {
        auto endAxis = viewport.unprojectPixelRay( Vector2f( viewportEnd.x, viewportEnd.y ) );
        if ( auto crossPL = intersection( referencePlane_, endAxis ) )
            worldEndPoint = *crossPL;
        else
            spdlog::warn( "Bad cross end axis and rotation plane" );

        const Vector3f vectorStart = worldStartPoint_ - xfCenterPoint_;
        const Vector3f vectorEnd = worldEndPoint - xfCenterPoint_;
        scale_ = vectorStart.lengthSq() < 1.0e-7f ? 1.0f :
            std::clamp( std::sqrt( vectorEnd.lengthSq() / vectorStart.lengthSq() ), 0.01f, 100.0f );

        setVisualizeVectors_( { xfCenterPoint_, worldEndPoint } );

        // Scale around center point (e.g. bounding box center)
        if ( transformMode_ == TransformMode::UniformScale )
        {
            currentXf_ = AffineXf3f::xfAround( Matrix3f::scale( scale_ ), xfCenterPoint_ );
        }
        else// if ( transformMode_ == TransformMode::NonUniformScale )
        {
            auto rotMat = Matrix3f::rotation( vectorEnd, Vector3f::plusX() );
            auto scaleAlongMat = rotMat.inverse() * Matrix3f::scale( scale_, 1, 1 ) * rotMat;
            currentXf_ = AffineXf3f::xfAround( scaleAlongMat, xfCenterPoint_ );
        }
    }
    else // if ( transformMode_ == TransformMode::Translation )
    {
        shift_ = ( worldEndPoint - worldStartPoint_ ).length();
        setVisualizeVectors_( { worldStartPoint_, worldEndPoint } );

        currentXf_ = AffineXf3f::translation( worldEndPoint - worldStartPoint_ );
    }

    applyCurrentXf_();

    return true;
}

bool MoveObjectByMouseImpl::onMouseUp( MouseButton button, int /*modifiers*/ )
{
    if ( button != currentButton_ )
        return false;

    bool res = transformMode_ != TransformMode::None;
    cancel();
    return res;
}

bool MoveObjectByMouseImpl::isMoving() const
{
    return transformMode_ != TransformMode::None && xfChanged_;
}

void MoveObjectByMouseImpl::cancel()
{
    connections_.clear();
    if ( transformMode_ == TransformMode::None )
        return;
    clear_();
}

MoveObjectByMouseImpl::TransformMode MoveObjectByMouseImpl::pick_( MouseButton button, int modifiers )
{
    // Use LMB for `Translation` and Ctrl+LMB for `Rotation`
    auto mode = modeFromPick_( button, modifiers );
    if ( mode == TransformMode::None )
        return mode;

    auto objPick = pickObjects_( objects_, modifiers );

    if ( objects_.empty() )
        return TransformMode::None;

    setCenterPoint_( objects_, xfCenterPoint_ );

    setStartPoint_( objPick, worldStartPoint_ );

    onPick_( mode, objects_, xfCenterPoint_, worldStartPoint_ );

    return mode;
}

void MoveObjectByMouseImpl::onPick_( TransformMode, const std::vector<std::shared_ptr<Object>>&, const Vector3f&, const Vector3f& )
{
}

ObjAndPick MoveObjectByMouseImpl::pickObjects_( std::vector<std::shared_ptr<Object>>& objects, int /*modifiers*/ ) const
{
    Viewer& viewer = getViewerInstance();
    Viewport& viewport = viewer.viewport( viewer.getHoveredViewportId() );
    // Pick a single object under cursor
    ObjAndPick res = viewport.pickRenderObject();
    const auto& [obj, pick] = res;
    if ( !obj || obj->isAncillary() )
    {
        objects = {};
        return res;
    }
    objects = { obj };
    return res;
}

MoveObjectByMouseImpl::TransformMode MoveObjectByMouseImpl::modeFromPickModifiers_( int modifiers ) const
{
    if ( modifiers == 0 )
        return TransformMode::Translation;
    else if ( modifiers == GLFW_MOD_CONTROL )
        return TransformMode::Rotation;
    return TransformMode::None;
}

MoveObjectByMouseImpl::TransformMode MoveObjectByMouseImpl::modeFromPick_( MouseButton button, int modifiers ) const
{
    auto mode = modeFromPickModifiers_( modifiers );
    if ( mode == TransformMode::None )
        return mode;
    if ( mode == TransformMode::UniformScale && button == MouseButton::Right )
        return TransformMode::NonUniformScale;
    if ( button == MouseButton::Left )
        return mode;
    return TransformMode::None;
}

void MoveObjectByMouseImpl::setStartPoint_( const ObjAndPick& objPick, Vector3f& startPoint ) const
{
    const auto& [obj, pick] = objPick;
    if ( !obj )
        return;
    startPoint = obj->worldXf()( pick.point );

    // Sample code to calculate reasonable startPoint when pick is unavailable
    // Vector2i mousePos = viewer.mouseController().getMousePos();
    // Vector3f viewportPos = viewer.screenToViewport( Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f ), viewport.id );
    // startPoint = viewport.unprojectPixelRay( Vector2f( viewportPos.x, viewportPos.y ) ).project( startPoint );
}

void MoveObjectByMouseImpl::setCenterPoint_( const std::vector<std::shared_ptr<Object>>& objects, Vector3f& centerPoint ) const
{
    Box3f box = getBbox_( objects );
    centerPoint = box.valid() ? box.center() : Vector3f{};
}

Box3f MoveObjectByMouseImpl::getBbox_( const std::vector<std::shared_ptr<Object>>& objects ) const
{
    Box3f worldBbox;
    for ( const std::shared_ptr<Object>& obj : objects )
        if ( obj )
            worldBbox.include( obj->getWorldBox() );
    return worldBbox;
}

void MoveObjectByMouseImpl::clear_()
{
    xfChanged_ = false;
    transformMode_ = TransformMode::None;
    objects_.clear();
    initialXfs_.clear();
    visualizeVectors_.clear();
    currentButton_ = MouseButton::NoButton;
}

void MoveObjectByMouseImpl::applyCurrentXf_()
{
    const bool appendHistory = historyEnabled_ && !xfChanged_;
    std::unique_ptr<ScopeHistory> scope = appendHistory ? std::make_unique<ScopeHistory>( "Move Object" ) : nullptr;
    auto itXf = initialXfs_.begin();
    changingXfFromMouseMove_ = true;
    for ( std::shared_ptr<Object>& obj : objects_ )
    {
        if ( appendHistory )
            AppendHistory<ChangeXfAction>( obj->name(), obj );
        obj->setWorldXf( currentXf_ * *itXf++ );
    }
    changingXfFromMouseMove_ = false;
    xfChanged_ = true;
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
