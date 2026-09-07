#include <MRViewer/MRRibbonRegisterItem.h>
#include <MRViewer/MRViewport.h>

namespace MR
{

class InvertViewMenuItem : public RibbonMenuItem
{
public:
   InvertViewMenuItem() : RibbonMenuItem( "Invert View" ) {}
   virtual bool action() override;
};

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
