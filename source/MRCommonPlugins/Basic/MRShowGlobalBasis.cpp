#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRObjectMesh.h"

namespace MR
{
class ShowGlobalBasisMenuItem : public RibbonMenuItem
{
public:
    ShowGlobalBasisMenuItem();
    virtual bool action() override;
};

ShowGlobalBasisMenuItem::ShowGlobalBasisMenuItem() :
    RibbonMenuItem( "Show_Hide Global Basis" )
{}

bool ShowGlobalBasisMenuItem::action()
{
    auto& viewer = Viewer::instanceRef();
    viewer.globalBasisAxes->setVisibilityMask( viewer.globalBasisAxes->visibilityMask() ^ viewer.viewport().id );
    return false;
}

MR_REGISTER_RIBBON_ITEM( ShowGlobalBasisMenuItem )
}