#include "MRSceneSettings.h"

namespace MR
{

namespace
{

struct SettingsState
{
    std::array<bool, size_t( SceneSettings::BoolType::Count ) > boolSettings;
    std::array<float, size_t( SceneSettings::FloatType::Count ) > floatSettings;

    SceneSettings::ShadingMode defaultShadingMode = SceneSettings::ShadingMode::AutoDetect;
    CNCMachineSettings cncMachineSettings;

    SettingsState()
    {
        using namespace SceneSettings;

        boolSettings[int( BoolType::UseDefaultScenePropertiesOnDeserialization )] = true;

        floatSettings[int( FloatType::FeaturePointsAlpha )] = 1;
        floatSettings[int( FloatType::FeatureLinesAlpha )] = 1;
        floatSettings[int( FloatType::FeatureMeshAlpha )] = 0.5f;
        floatSettings[int( FloatType::FeatureSubPointsAlpha )] = 1;
        floatSettings[int( FloatType::FeatureSubLinesAlpha )] = 1;
        floatSettings[int( FloatType::FeatureSubMeshAlpha )] = 0.5f;
        floatSettings[int( FloatType::FeaturePointSize )] = 10;
        floatSettings[int( FloatType::FeatureSubPointSize )] = 8;
        floatSettings[int( FloatType::FeatureLineWidth )] = 3;
        floatSettings[int( FloatType::FeatureSubLineWidth )] = 2;
        floatSettings[int( FloatType::AmbientCoefSelectedObj )] = 2.5f;
    }
};

SettingsState &GetSettingsState()
{
    static SettingsState ret;
    return ret;
}

} // namespace

void SceneSettings::reset()
{
    GetSettingsState() = {};
}

bool SceneSettings::get( BoolType type )
{
    return GetSettingsState().boolSettings[int( type )];
}

float SceneSettings::get( FloatType type )
{
    return GetSettingsState().floatSettings[int( type )];
}

void SceneSettings::set( BoolType type, bool value )
{
    GetSettingsState().boolSettings[int( type )] = value;
}

void SceneSettings::set( FloatType type, float value )
{
    GetSettingsState().floatSettings[int( type )] = value;
}

SceneSettings::ShadingMode SceneSettings::getDefaultShadingMode()
{
    return GetSettingsState().defaultShadingMode;
}

void SceneSettings::setDefaultShadingMode( SceneSettings::ShadingMode mode )
{
    GetSettingsState().defaultShadingMode = mode;
}

const CNCMachineSettings& SceneSettings::getCNCMachineSettings()
{
    return GetSettingsState().cncMachineSettings;
}

void SceneSettings::setCNCMachineSettings( const CNCMachineSettings& settings )
{
    GetSettingsState().cncMachineSettings = settings;
}

}
