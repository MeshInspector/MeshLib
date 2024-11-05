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
    if ( transformMode_ == TransformMode::Scale )
        ImGui::SetTooltip( "Scale : %s", valueToString<RatioUnit>( scale_ ).c_str() );
}

bool MoveObjectByMouseImpl::onMouseDown( MouseButton button, int modifiers )
{
    Viewer& viewer = getViewerInstance();
    Viewport& viewport = viewer.viewport();

    cancel();
    transformMode_ = pick_( button, modifiers, objects_, xfCenterPoint_, worldStartPoint_ );
    if ( transformMode_ == TransformMode::None )
    {
        clear_();
        return false;
    }

    currentButton_ = button;
    screenStartPoint_ = minDistance() > 0 ? viewer.mouseController().getMousePos() : cNoPoint;
    angle_ = 0.f;
    shift_ = 0.f;
    scale_ = 1.f;
    currentXf_ = {};
    viewportStartPointZ_ = viewport.projectToViewportSpace( worldStartPoint_ ).z;
    initialXfs_.clear();
    for ( std::shared_ptr<Object>& obj : objects_ )
        initialXfs_.push_back( obj->worldXf() );

    if ( transformMode_ == TransformMode::Rotation )
    {
        Vector3f viewportCenterPoint = viewport.projectToViewportSpace( xfCenterPoint_ );

        Line3f centerAxis = viewport.unprojectPixelRay( Vector2f( viewportCenterPoint.x, viewportCenterPoint.y ) );
        referencePlane_ = Plane3f::fromDirAndPt( centerAxis.d.normalized(), xfCenterPoint_ );

        Vector3f viewportStartPoint = viewport.projectToViewportSpace( worldStartPoint_ );
        Line3f startAxis = viewport.unprojectPixelRay( Vector2f( viewportStartPoint.x, viewportStartPoint.y ) );

        if ( auto crossPL = intersection( referencePlane_, startAxis ) )
            worldStartPoint_ = *crossPL;
        else
            spdlog::warn( "Bad cross start axis and rotation plane" );

        setVisualizeVectors_( { xfCenterPoint_, worldStartPoint_, xfCenterPoint_, worldStartPoint_ } );
    }
    else if ( transformMode_ == TransformMode::Scale )
    {
        Vector3f viewportCenterPoint = viewport.projectToViewportSpace( xfCenterPoint_ );

        Line3f centerAxis = viewport.unprojectPixelRay( Vector2f( viewportCenterPoint.x, viewportCenterPoint.y ) );
        referencePlane_ = Plane3f::fromDirAndPt( centerAxis.d.normalized(), xfCenterPoint_ );

        Vector3f viewportStartPoint = viewport.projectToViewportSpace( worldStartPoint_ );
        Line3f startAxis = viewport.unprojectPixelRay( Vector2f( viewportStartPoint.x, viewportStartPoint.y ) );

        if ( auto crossPL = intersection( referencePlane_, startAxis ) )
            worldStartPoint_ = *crossPL;
        else
            spdlog::warn( "Bad cross start axis and rotation plane" );
        setVisualizeVectors_( { xfCenterPoint_, worldStartPoint_ } );
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

    if ( screenStartPoint_ != cNoPoint &&
         ( screenStartPoint_ - viewer.mouseController().getMousePos() ).lengthSq() <
             minDistance() * minDistance() )
        return true;
    screenStartPoint_ = cNoPoint;

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
        AffineXf3f rotation = AffineXf3f::linear( Matrix3f::rotation( vectorStart, worldEndPoint - xfCenterPoint_ ) );
        AffineXf3f toCenterPoint = AffineXf3f::translation( xfCenterPoint_ );
        currentXf_ = toCenterPoint * rotation * toCenterPoint.inverse();
    }
    else if ( transformMode_ == TransformMode::Scale )
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
        AffineXf3f toCenterPoint = AffineXf3f::translation( xfCenterPoint_ );
        currentXf_ = toCenterPoint * AffineXf3f::linear( Matrix3f::scale( scale_ ) ) * toCenterPoint.inverse();
    }
    else // if ( transformMode_ == TransformMode::Translation )
    {
        shift_ = ( worldEndPoint - worldStartPoint_ ).length();
        setVisualizeVectors_( { worldStartPoint_, worldEndPoint } );

        currentXf_ = AffineXf3f::translation( worldEndPoint - worldStartPoint_ );

        // Clamp movement
        Box3f worldBox = getBbox_( objects_ );
        float minSizeDim = worldBox.valid() ? worldBox.size().length() : 0;
        if ( minSizeDim == 0 )
            minSizeDim = 1.f;
        for ( const AffineXf3f &xf : initialXfs_ )
            for ( auto i = 0; i < 3; i++ )
                // ( currentXf_ * initialXf_ ).b[i] must be in -/+ cMaxTranslationMultiplier * minSizeDim
                currentXf_.b[i] = std::clamp( currentXf_.b[i],
                     -xf.b[i] - cMaxTranslationMultiplier * minSizeDim,
                     -xf.b[i] + cMaxTranslationMultiplier * minSizeDim );
    }

    applyCurrentXf_( false );

    return true;
}

bool MoveObjectByMouseImpl::onMouseUp( MouseButton button, int /*modifiers*/ )
{
    if ( transformMode_ == TransformMode::None || button != currentButton_ )
        return false;

    if ( screenStartPoint_ != cNoPoint )
    {
        clear_();
        return false;
    }

    resetXfs_();
    applyCurrentXf_( true );
    clear_();

    return true;
}

bool MoveObjectByMouseImpl::isMoving() const
{
    return transformMode_ != TransformMode::None && screenStartPoint_ == cNoPoint;
}

void MoveObjectByMouseImpl::cancel()
{
    if ( transformMode_ == TransformMode::None )
        return;
    resetXfs_();
    clear_();
}

MRVIEWER_API MoveObjectByMouseImpl::TransformMode MoveObjectByMouseImpl::pick_( MouseButton button, int modifiers,
    std::vector<std::shared_ptr<Object>>& objects, Vector3f& centerPoint, Vector3f& startPoint )
{
    // Use LMB for `Translation` and Ctrl+LMB for `Rotation`
    if ( !( button == MouseButton::Left && ( modifiers == 0 || modifiers == GLFW_MOD_CONTROL ) ) )
        return TransformMode::None;

    Viewer& viewer = getViewerInstance();
    Viewport& viewport = viewer.viewport();

    // Pick a single object under cursor
    auto [obj, pick] = viewport.pickRenderObject();

    // Check if picked something, and is not an ancillary object
    if ( !obj || obj->isAncillary() )
        return TransformMode::None;

    // Use bounding box center as transform center and the picked point as an initial position
    objects = { obj };
    Box3f box = getBbox_( objects );
    centerPoint = box.valid() ? box.center() : Vector3f{};
    startPoint = obj->worldXf()( pick.point );
    // Sample code to calculate reasonable startPoint when pick is unavailable
    // Vector2i mousePos = viewer.mouseController().getMousePos();
    // Vector3f viewportPos = viewer.screenToViewport( Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f ), viewport.id );
    // startPoint = viewport.unprojectPixelRay( Vector2f( viewportPos.x, viewportPos.y ) ).project( startPoint );
    return modifiers == 0 ? TransformMode::Translation : TransformMode::Rotation;
}

Box3f MoveObjectByMouseImpl::getBbox_( const std::vector<std::shared_ptr<Object>>& objects )
{
    Box3f worldBbox;
    for ( const std::shared_ptr<Object>& obj : objects )
        if ( obj )
            worldBbox.include( obj->getWorldBox() );
    return worldBbox;
}

void MoveObjectByMouseImpl::clear_()
{
    transformMode_ = TransformMode::None;
    objects_.clear();
    initialXfs_.clear();
    visualizeVectors_.clear();
    currentButton_ = MouseButton::NoButton;
}

void MoveObjectByMouseImpl::applyCurrentXf_( bool history )
{
    std::unique_ptr<ScopeHistory> scope = history ? std::make_unique<ScopeHistory>( "Move Object" ) : nullptr;
    auto itXf = initialXfs_.begin();
    for ( std::shared_ptr<Object>& obj : objects_ )
    {
        if ( history )
            AppendHistory<ChangeXfAction>( "xf", obj );
        obj->setWorldXf( currentXf_ * *itXf++ );
    }
}

void MoveObjectByMouseImpl::resetXfs_()
{
    auto itXf = initialXfs_.begin();
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
