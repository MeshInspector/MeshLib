#include "MRViewer/MRRibbonRegisterItem.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRViewer/MRViewportGlobalBasis.h"
#include "MRCommonPlugins/Basic/MRDrawViewportWidgetsItem.h"

namespace MR
{
class ShowGlobalBasisMenuItem : public RibbonMenuItem, public ProvidesViewportWidget
{
public:
    ShowGlobalBasisMenuItem();
    bool action() override;

    void providedViewportWidgets( ViewportWidgetInterface& in ) override;
};

ShowGlobalBasisMenuItem::ShowGlobalBasisMenuItem() :
    RibbonMenuItem( "Show_Hide Global Basis" )
{}

bool ShowGlobalBasisMenuItem::action()
{
    auto& viewer = Viewer::instanceRef();
    auto vpid = viewer.viewport().id;
    bool isVisible = viewer.globalBasis->isVisible( vpid );
    bool isGridVisible = viewer.globalBasis->isGridVisible( vpid );
    if ( !isVisible )
    {
        viewer.globalBasis->setVisible( true, vpid );
        viewer.globalBasis->setGridVisible( false, vpid );
    }
    else if ( !isGridVisible )
    {
        viewer.globalBasis->setGridVisible( true, vpid );
    }
    else
    {
        viewer.globalBasis->setVisible( false, vpid );
        viewer.globalBasis->setGridVisible( false, vpid );
    }
    return false;
}

void ShowGlobalBasisMenuItem::providedViewportWidgets( ViewportWidgetInterface& in )
{
    in.addButton( 20, "Toggle Basis", false, "Viewport basis",
        [this]{ action(); }
    );
}

MR_REGISTER_RIBBON_ITEM( ShowGlobalBasisMenuItem )
}
