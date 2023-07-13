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

const CNCMachineSettings& SceneSettings::getCNCMachineSettings()
{
    return instance_().cncMachineSettings_;
}

void SceneSettings::setCNCMachineSettings( const CNCMachineSettings& settings )
{
    instance_().cncMachineSettings_ = settings;
}

SceneSettings& SceneSettings::instance_()
{
    static SceneSettings instance;
    return instance;
}

}