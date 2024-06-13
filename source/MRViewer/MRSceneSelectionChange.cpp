#include "MRSceneSelectionChange.h"
#include "MRStatePlugin.h"
#include <MRViewer/MRCommandLoop.h>
namespace MR
{

void SceneSelectionChangeClose::updateSelection( const std::vector<std::shared_ptr<const Object>>& )
{
    auto thisPlugin = dynamic_cast< StateBasePlugin* >( this );
    thisPlugin->enable( false );
}

void SceneSelectionChangeRestart::updateSelection( const std::vector<std::shared_ptr<const Object>>& objects )
{
    //in the loop, the plugin is passed through and removed from the vector, and after that it should not be added again
    CommandLoop::appendCommand( [this, objects_ = objects] ()
    {
        auto thisPlugin = dynamic_cast< StateBasePlugin* >( this );
        if ( !thisPlugin->enable( false ) )
            return;
        if ( !thisPlugin->isAvailable( objects_ ).empty() )
            return;
        thisPlugin->enable( true );
    }
    );
}

}
