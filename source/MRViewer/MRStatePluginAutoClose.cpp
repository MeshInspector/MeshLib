#include "MRStatePluginAutoClose.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObject.h"

namespace MR
{

void PluginCloseOnSelectedObjectRemove::onPluginEnable_()
{
    selectedObjs_ = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
}

void PluginCloseOnSelectedObjectRemove::onPluginDisable_()
{
    selectedObjs_.clear();
}

bool PluginCloseOnSelectedObjectRemove::shouldClose_() const
{
    for ( const auto& obj : selectedObjs_ )
    {
        if ( !obj->isAncestor( &SceneRoot::get() ) )
            return true;
    }
    return false;
}

}