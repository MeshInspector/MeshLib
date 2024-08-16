#include "MRMoveObjectByMouse.h"
#include "MRViewer/MRMouse.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRRibbonSchema.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRMouseController.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRIntersection.h"
#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRSceneColors.h"
#include "MRPch/MRSpdlog.h"

namespace
{
MR::MoveObjectByMouse* sInstance{ nullptr };
}

namespace MR
{

MoveObjectByMouse::MoveObjectByMouse() :
    PluginParent( "Move object", StatePluginTabs::Basic )
{
    sInstance = this;
}

MoveObjectByMouse::~MoveObjectByMouse()
{
    sInstance = nullptr;
}

MoveObjectByMouse* MoveObjectByMouse::instance()
{
    return sInstance;
}

bool MoveObjectByMouse::onDisable_()
{
    moveByMouse_.cancel();
    return true;
}

void MoveObjectByMouse::drawDialog( float menuScaling, ImGuiContext*)
{
    auto menuWidth = 400.f * menuScaling;
    if ( !ImGuiBeginWindow_( { .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    ImGui::Text( "%s", "Click and hold LMB on object to move" );
    ImGui::Text( "%s", "Click CTRL + LMB and hold LMB on object to rotate" );
    ImGui::Text( "%s", "Click ALT + LMB and hold LMB on object to scale" );
    ImGui::Text( "%s", "Press Shift to move selected objects together" );

    moveByMouse_.onDrawDialog( menuScaling );

    ImGui::EndCustomStatePlugin();
}

bool MoveObjectByMouse::onDragStart_( MouseButton btn, int modifiers )
{
    return moveByMouse_.onMouseDown( btn, modifiers );
}

bool MoveObjectByMouse::onDrag_( int x, int y )
{
    viewer->select_hovered_viewport();
    return moveByMouse_.onMouseMove( x, y );
}

bool MoveObjectByMouse::onDragEnd_( MouseButton btn, int modifiers )
{
    return moveByMouse_.onMouseUp( btn, modifiers );
}

MoveObjectByMouseImpl::TransformMode MoveObjectByMouse::MoveObjectByMouseWithSelected::pick_( MouseButton button, int modifiers,
    std::vector<std::shared_ptr<Object>>& objects, Vector3f& centerPoint, Vector3f& startPoint )
{
    if ( button != MouseButton::Left || ( modifiers & ~( GLFW_MOD_SHIFT | GLFW_MOD_CONTROL | GLFW_MOD_ALT ) ) != 0 ||
         ( modifiers & ( GLFW_MOD_CONTROL | GLFW_MOD_ALT ) ) == ( GLFW_MOD_CONTROL | GLFW_MOD_ALT ) )
        return TransformMode::None;

    Viewer& viewerInstance = getViewerInstance();
    Viewport& viewport = viewerInstance.viewport();

    // Pick non-ancillary object
    auto [obj, pick] = viewport.pickRenderObject();
    if ( obj && obj->isAncillary() )
        obj = nullptr;

    if ( ( modifiers & GLFW_MOD_SHIFT ) == 0 )
    {
        // Move picked object
        if ( !obj )
            return TransformMode::None;
        objects = { obj };
    }
    else
    {
        // Move selected objects regardless of pick
        objects = getAllObjectsInTree<Object>( SceneRoot::get(), ObjectSelectivityType::Selected );
        if ( std::find( objects.begin(), objects.end(), obj ) == objects.end() )
            obj = nullptr; // Use picked object only if it is selected
    }

    // See MoveObjectByMouseImpl::pick_
    if ( obj )
    {
        startPoint = obj->worldXf()( pick.point );
    }
    else
    {
        Vector2i mousePos = viewerInstance.mouseController().getMousePos();
        Vector3f viewportPos = viewerInstance.screenToViewport( Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f ), viewport.id );
        startPoint = viewport.unprojectPixelRay( Vector2f( viewportPos.x, viewportPos.y ) ).project( startPoint );
    }
    Box3f box = getBbox_( objects );
    centerPoint = box.valid() ? box.center() : Vector3f{};

    return ( modifiers & GLFW_MOD_CONTROL ) != 0 ? TransformMode::Rotation : 
        ( modifiers & GLFW_MOD_ALT ) != 0 ? TransformMode::Scale : TransformMode::Translation;
}

MR_REGISTER_RIBBON_ITEM( MoveObjectByMouse )

}
