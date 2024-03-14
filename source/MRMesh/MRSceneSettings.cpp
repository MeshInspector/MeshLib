#include "MRSceneSettings.h"

namespace MR
{

bool SceneSettings::get( BoolType type )
{
    return instance_().boolSettings_[int( type )];
}

float SceneSettings::get( FloatType type )
{
    return instance_().floatSettings_[int( type )];
}

void SceneSettings::set( BoolType type, bool value )
{
    instance_().boolSettings_[int( type )] = value;
}

void SceneSettings::set( FloatType type, float value )
{
    instance_().floatSettings_[int( type )] = value;
}

SceneSettings::ShadingMode SceneSettings::getDefaultShadingMode()
{
    return instance_().defaultShadingMode_;
}

void SceneSettings::setDefaultShadingMode( SceneSettings::ShadingMode mode )
{
    instance_().defaultShadingMode_ = mode;
}

const CNCMachineSettings& SceneSettings::getCNCMachineSettings()
{
    return instance_().cncMachineSettings_;
}

void SceneSettings::setCNCMachineSettings( const CNCMachineSettings& settings )
{
    instance_().cncMachineSettings_ = settings;
}

SceneSettings::SceneSettings()
{
    boolSettings_[int( BoolType::UseDefaultScenePropertiesOnDeserialization )] = true;

    floatSettings_[int( FloatType::FeatureMeshAlpha )] = 0.5f;
    floatSettings_[int( FloatType::FeaturePointSize )] = 10;
    floatSettings_[int( FloatType::FeatureSubPointSize )] = 8;
    floatSettings_[int( FloatType::FeatureLineWidth )] = 3;
    floatSettings_[int( FloatType::FeatureSubLineWidth )] = 2;
}

SceneSettings& SceneSettings::instance_()
{
    static SceneSettings instance;
    return instance;
}

}
