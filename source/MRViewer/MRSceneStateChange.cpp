#include "MRSceneStateChange.h"
#include "MRStatePlugin.h"

namespace MR
{

void SceneStateChangeClose::updateSelection( const std::vector<std::shared_ptr<const Object>>& )
{
    auto thisPlugin = dynamic_cast< StateBasePlugin* >( this );
    thisPlugin->enable( false );
}

void SceneStateChangeRestart::updateSelection( const std::vector<std::shared_ptr<const Object>>& objects )
{
    auto thisPlugin = dynamic_cast< StateBasePlugin* >( this );
    if ( !thisPlugin->enable( false ) )
        return;
    if ( !thisPlugin->isAvailable( objects ).empty() )
        return;
    thisPlugin->enable( true );
}

}