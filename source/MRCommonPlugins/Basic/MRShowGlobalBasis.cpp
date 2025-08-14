#include "MRViewer/MRRibbonRegisterItem.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRViewer/MRViewportGlobalBasis.h"

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
    auto vpid = viewer.viewport().id;
    viewer.globalBasis->setVisible( !viewer.globalBasis->isVisible( vpid ), vpid );
    return false;
}

MR_REGISTER_RIBBON_ITEM( ShowGlobalBasisMenuItem )
}