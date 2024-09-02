#include "MRMoveObjectByMouse.h"
#include "MRViewer/MRMouse.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRRibbonSchema.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRMouseController.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/ImGuiMenu.h"
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

    ImGui::Text( "Click and hold LMB to move or transform" );

    ImGui::Separator();

    ImGui::Text( "Mode:" );
    UI::radioButtonOrModifier( "Move",   moveByMouse_.modXfMode, int( XfMode::Move ),   0,              ImGuiMod_Ctrl | ImGuiMod_Alt );
    ImGui::SameLine();
    UI::radioButtonOrModifier( "Rotate", moveByMouse_.modXfMode, int( XfMode::Rotate ), ImGuiMod_Ctrl,  ImGuiMod_Ctrl | ImGuiMod_Alt );
    ImGui::SameLine();
    UI::radioButtonOrModifier( "Scale",  moveByMouse_.modXfMode, int( XfMode::Scale ),  ImGuiMod_Alt,   ImGuiMod_Ctrl | ImGuiMod_Alt );

    ImGui::Text( "Target:" );
    UI::radioButtonOrModifier( "Picked object",      moveByMouse_.modXfTarget, int( XfTarget::Picked ),                0, ImGuiMod_Shift );
    ImGui::SameLine();
    UI::radioButtonOrModifier( "Selected object(s)", moveByMouse_.modXfTarget, int( XfTarget::Selected ), ImGuiMod_Shift, ImGuiMod_Shift );

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

void MoveObjectByMouse::postDraw_()
{
    if (const auto& menu = getViewerInstance().getMenuPlugin() )
        moveByMouse_.onDrawDialog( menu->menu_scaling() );
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

    if ( int( modXfTarget ) == int( XfTarget::Selected ) || ( modifiers & GLFW_MOD_SHIFT ) == GLFW_MOD_SHIFT )
    {
        // Move selected objects regardless of pick
        objects = getAllObjectsInTree<Object>( SceneRoot::get(), ObjectSelectivityType::Selected );
        if ( std::find( objects.begin(), objects.end(), obj ) == objects.end() )
            obj = nullptr; // Use picked object only if it is selected
    }
    else
    {
        // Move picked object
        if ( !obj )
            return TransformMode::None;
        objects = { obj };
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

    if ( int( modXfMode ) == int( XfMode::Scale ) || ( modifiers & GLFW_MOD_ALT ) == GLFW_MOD_ALT )
        return TransformMode::Scale;
    else if ( int( modXfMode ) == int( XfMode::Rotate ) || ( modifiers & GLFW_MOD_CONTROL ) == GLFW_MOD_CONTROL )
        return TransformMode::Rotation;
    else
        return TransformMode::Translation;
}

MR_REGISTER_RIBBON_ITEM( MoveObjectByMouse )

}
