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

    void toggleInViewport( ViewportId id );

    void providedViewportWidgets( ViewportWidgetInterface& in ) override;

    ViewportMask showButtonInViewports = ViewportMask::all();
};

ShowGlobalBasisMenuItem::ShowGlobalBasisMenuItem() :
    RibbonMenuItem( "Show_Hide Global Basis" )
{}

void ShowGlobalBasisMenuItem::toggleInViewport( ViewportId id )
{
    auto& viewer = Viewer::instanceRef();
    bool isVisible = viewer.globalBasis->isVisible( id );
    bool isGridVisible = viewer.globalBasis->isGridVisible( id );
    if ( !isVisible )
    {
        viewer.globalBasis->setVisible( true, id );
        viewer.globalBasis->setGridVisible( false, id );
    }
    else if ( !isGridVisible )
    {
        viewer.globalBasis->setGridVisible( true, id );
    }
    else
    {
        viewer.globalBasis->setVisible( false, id );
        viewer.globalBasis->setGridVisible( false, id );
    }
}

bool ShowGlobalBasisMenuItem::action()
{
    toggleInViewport( getViewerInstance().viewport().id );
    return false;
}

void ShowGlobalBasisMenuItem::providedViewportWidgets( ViewportWidgetInterface& in )
{
    auto id = in.viewportId();
    if ( !showButtonInViewports.contains( id ) )
        return;

    in.addButton( 20, "Toggle Basis", false, "Viewport basis",
        [this, id]{ toggleInViewport( id ); }
    );
}

MR_REGISTER_RIBBON_ITEM( ShowGlobalBasisMenuItem )
}
