#include "MRSelectObjectByClick.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MR2to3.h"
#include <GLFW/glfw3.h>

namespace MR
{

SelectObjectByClick::SelectObjectByClick() :
    PluginParent( "Select objects", StatePluginTabs::Selection )
{
}

void SelectObjectByClick::drawDialog( float, ImGuiContext* )
{
    if ( !picked_ )
        return;
    auto drawList = ImGui::GetBackgroundDrawList();
    auto downPos = getViewerInstance().mouseController.getDownMousePos();
    auto currPos = getViewerInstance().mouseController.getMousePos();
    Box2i rect;
    rect.include( downPos );
    rect.include( currPos );
    drawList->AddRect( ImVec2( float( rect.min.x ), float( rect.min.y ) ), ImVec2( float( rect.max.x ), float( rect.max.y ) ),
                       Color::white().getUInt32() );
}

bool SelectObjectByClick::onMouseDown_( MouseButton button, int modifier )
{
    if ( button != MouseButton::Left )
        return false;

    picked_ = true;
    ctrl_ = modifier == GLFW_MOD_CONTROL;
    return true;
}

bool SelectObjectByClick::onMouseUp_( MouseButton button, int )
{
    if ( !picked_ || button != MouseButton::Left )
        return false;
    select_( true );
    picked_ = false;
    ctrl_ = false;
    return true;
}

bool SelectObjectByClick::onMouseMove_( int, int )
{
    if ( !picked_ )
        return false;

    select_( false );

    return true;
}

void SelectObjectByClick::select_( bool up )
{
    auto downPos = getViewerInstance().mouseController.getDownMousePos();
    auto currPos = getViewerInstance().mouseController.getMousePos();

    std::vector<std::shared_ptr<VisualObject>> newSelection;
    const auto& viewport = viewer->viewport();
    bool smallPick = ( downPos - currPos ).lengthSq() < 9;
    if ( smallPick ) // 3*3
    {
        const auto [obj, pick] = viewport.pick_render_object();
        if ( obj )
            newSelection = { obj };
    }
    else
    {
        Box2i rect;
        rect.include( Vector2i( to2dim( viewer->screenToViewport( to3dim( Vector2f( downPos ) ), viewport.id ) ) ) );
        rect.include( Vector2i( to2dim( viewer->screenToViewport( to3dim( Vector2f( currPos ) ), viewport.id ) ) ) );
        newSelection = getViewerInstance().viewport().findObjectsInRect( rect );
    }

    if ( up && smallPick && ctrl_ )
    {
        for ( auto obj : newSelection )
            obj->select( !obj->isSelected() );
    }
    else
    {
        auto selectedObjects = getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& object : selectedObjects )
            object->select( false );

        for ( auto obj : newSelection )
            obj->select( true );
    }
}

MR_REGISTER_RIBBON_ITEM( SelectObjectByClick )

}
