#include "MRMoveObjectByMouse.h"
#include "MRViewer/MRMouse.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRSceneColors.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRIntersection.h"
#include "MRMesh/MRVisualObject.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/ImGuiHelpers.h"

namespace MR
{

MoveObjectByMouse::MoveObjectByMouse() :
    PluginParent( "Move object", StatePluginTabs::Basic )
{
}

void MoveObjectByMouse::drawDialog( float menuScaling, ImGuiContext*)
{
    auto menuWidth = 400.f * menuScaling;
    if ( !ImGuiBeginWindow_( { .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    ImGui::Text( "%s", "Click and hold LMB on object to move" );
    ImGui::Text( "%s", "Click CTRL + LMB and hold LMB on object to rotate" );

    moveByMouse_.onDrawDialog( menuScaling );

    ImGui::EndCustomStatePlugin();
}

bool MoveObjectByMouse::onMouseDown_( MouseButton btn, int modifiers )
{
    return moveByMouse_.onMouseDown( btn, modifiers );
}

bool MoveObjectByMouse::onMouseMove_( int x, int y )
{
    return moveByMouse_.onMouseMove( x, y );
}

bool MoveObjectByMouse::onMouseUp_( MouseButton btn, int modifiers )
{
    return moveByMouse_.onMouseUp( btn, modifiers );
}

MR_REGISTER_RIBBON_ITEM( MoveObjectByMouse )

}
