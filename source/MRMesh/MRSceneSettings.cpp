#include "MRSceneSettings.h"

namespace MR
{

bool SceneSettings::get( Type type )
{
    return instance_().settings_[int( type )];
}

void SceneSettings::set( Type type, bool value )
{
    instance_().settings_[int( type )] = value;
}

SceneSettings& SceneSettings::instance_()
{
    static SceneSettings instance;
    return instance;
}

}