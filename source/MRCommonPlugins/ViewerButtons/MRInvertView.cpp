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
   virtual std::optional<Shortcut> defaultShortcut_( const ShortcutConfig& conf ) const override;
};

std::optional<Shortcut> InvertViewMenuItem::defaultShortcut_( const ShortcutConfig& conf ) const
{
    if ( !conf.allowBase )
        return {};
    return Shortcut{ { GLFW_KEY_KP_9, 0 }, ShortcutCategory::View };
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
