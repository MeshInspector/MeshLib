#include "MRMoveObjectByMouse.h"
#include "MRViewer/MRMouse.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRRibbonSchema.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRViewer/ImGuiHelpers.h"
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
    ImGui::Text( "%s", "Press Shift to move selected objects together" );

    moveByMouse_.onDrawDialog( menuScaling );

    ImGui::EndCustomStatePlugin();
}

bool MoveObjectByMouse::onDragStart_( MouseButton btn, int modifiers )
{
    if ( ( modifiers & ~( GLFW_MOD_SHIFT | GLFW_MOD_CONTROL ) ) != 0 )
        return false;
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

std::vector<std::shared_ptr<Object>> MoveObjectByMouse::MoveObjectByMouseWithSelected::getObjects_(
    const std::shared_ptr<VisualObject>& obj, const PointOnObject&, int modifiers )
{
    if ( ( modifiers & GLFW_MOD_SHIFT ) != 0 && obj->isSelected() )
        return getAllObjectsInTree<Object>( SceneRoot::get(), ObjectSelectivityType::Selected );
    return { obj };
}

MR_REGISTER_RIBBON_ITEM( MoveObjectByMouse )

}
