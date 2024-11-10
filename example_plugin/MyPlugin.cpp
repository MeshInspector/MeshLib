#include "MRViewer/MRRibbonMenuItem.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRRibbonSchema.h"

namespace ExamplePlugin
{
using namespace MR;

class MyTool : public RibbonMenuItem
{
public:
    MyTool() : RibbonMenuItem( "My Tool" ) {}
    virtual bool action() override
    {
        showModal( "Hello World!", MR::NotificationType::Info );
        return false; // true will lead Viewer to keep this tool activated in ribbon
    }
};

class MyStateTool : public StatePlugin
{
public:
    MyStateTool() : StatePlugin( "My State Tool" ) {}

    virtual void drawDialog( float menuScaling, ImGuiContext* ) override
    {
        if ( !ImGuiBeginWindow_( { .width = 200 * menuScaling } ) )
            return;
        UI::transparentTextWrapped( "Hello World!" );
        ImGui::EndCustomStatePlugin();
    }
};

MR_REGISTER_RIBBON_ITEM( MyTool )

MR_REGISTER_RIBBON_ITEM( MyStateTool )

}
