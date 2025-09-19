#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRUIStyle.h"

class HelloWorldPlugin : public MR::StatePlugin
{
public:
    HelloWorldPlugin():
        MR::StatePlugin( "Hello World" )
    {}
    virtual void drawDialog( ImGuiContext* ) override
    {
        if ( !ImGuiBeginWindow_( { .collapsed = &dialogIsCollapsed_,.width = 200.0f * MR::UI::scale() } ) )
            return;

        ImGui::TextWrapped( "Hello World!" );
        ImGui::EndCustomStatePlugin();
    }
};

MR_REGISTER_RIBBON_ITEM( HelloWorldPlugin )
