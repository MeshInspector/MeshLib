#include "MRViewer/MRRibbonSchema.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRRibbonButtonDrawer.h"
#include "MRViewer/MRStatePlugin.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectsAccess.h"

namespace MR
{

class ObjectInfo : public StatePlugin
{
public:
    ObjectInfo();

    virtual void drawDialog( float menuScaling, ImGuiContext* ) override;
    virtual bool blocking() const override { return false; }
};

ObjectInfo::ObjectInfo():
    StatePlugin( "Object Info", StatePluginTabs::Test )
{
}

void ObjectInfo::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 300 * menuScaling;
    if ( !ImGuiBeginWindow_( { .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    if ( auto obj = getDepthFirstObject<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
    {
        ImGui::Text( "Selected object: %s", obj->name().c_str() );
        for ( const auto & line : obj->getInfoLines() )
            ImGui::Text( "%s", line.c_str() );
    }
    else
    {
        ImGui::Text( "No object selected" );
    }

    ImGui::EndCustomStatePlugin();
}

MR_REGISTER_RIBBON_ITEM( ObjectInfo )

}
