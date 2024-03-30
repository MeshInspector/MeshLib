#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRRibbonMenu.h"

class HelloWorldPlugin : public MR::StatePlugin
{
public:
    HelloWorldPlugin():
        MR::StatePlugin( "Hello World" )
    {}
    virtual void drawDialog( float menuScaling, ImGuiContext* ) override
    {
        if ( !ImGuiBeginWindow_( { .collapsed = &dialogIsCollapsed_,.width = 200.0f * menuScaling,.menuScaling = menuScaling } ) )
            return;

        ImGui::TextWrapped( "Hello World!" );
        ImGui::EndCustomStatePlugin();
    }
};

MR_REGISTER_RIBBON_ITEM( HelloWorldPlugin )