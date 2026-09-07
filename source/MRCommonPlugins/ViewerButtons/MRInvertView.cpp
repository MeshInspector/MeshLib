#include <MRViewer/MRRibbonRegisterItem.h>
#include <MRViewer/MRViewport.h>
#include <MRViewer/MRShortcutManager.h>
#include <MRViewer/MRGladGlfw.h>

namespace MR
{

class InvertViewMenuItem : public RibbonMenuItem
{
public:
   InvertViewMenuItem() : RibbonMenuItem( "Invert View" ) {}
   virtual bool action() override;
   virtual void registerShortcut( RibbonMenu& menu, const ShortcutConfig& conf ) override;
};

void InvertViewMenuItem::registerShortcut( RibbonMenu& menu, const ShortcutConfig& conf )
{
    if ( conf.allowBase )
        registerShortcut_( menu, { GLFW_KEY_KP_9, 0 }, ShortcutCategory::View );
}

bool InvertViewMenuItem::action()
{
    auto& viewport = Viewport::get();
    auto up = viewport.getUpDirection();
    auto back = viewport.getBackwardDirection();

    viewport.cameraLookAlong( back, up );

    viewport.preciseFitDataToScreenBorder( { 0.9f } );
    return false;
}

MR_REGISTER_RIBBON_ITEM( InvertViewMenuItem )

} //namespace MR
