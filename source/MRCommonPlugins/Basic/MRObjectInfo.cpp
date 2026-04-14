#include "MRViewer/MRRibbonRegisterItem.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRRibbonButtonDrawer.h"
#include "MRViewer/MRStatePlugin.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRViewer/MRI18n.h"

namespace MR
{

class ObjectInfo : public StatePlugin
{
public:
    ObjectInfo();

    virtual void drawDialog( ImGuiContext* ) override;
    virtual bool blocking() const override { return false; }
};

ObjectInfo::ObjectInfo():
    StatePlugin( "Object Info", StatePluginTabs::Test )
{
}

void ObjectInfo::drawDialog( ImGuiContext* )
{
    auto menuWidth = 300 * UI::scale();
    if ( !ImGuiBeginWindow_( { .width = menuWidth } ) )
        return;

    if ( auto obj = getDepthFirstObject<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
    {
        ImGui::Text( "%s: %s", _tr( "Selected object" ), obj->name().c_str() );
        for ( const auto & line : obj->getInfoLines() )
            ImGui::Text( "%s", line.c_str() );
    }
    else
    {
        ImGui::Text( "%s", _tr( "No object selected" ) );
    }

    ImGui::EndCustomStatePlugin();
}

MR_REGISTER_RIBBON_ITEM( ObjectInfo )

}
