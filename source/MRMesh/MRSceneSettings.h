#pragma once
#include "MRMeshFwd.h"
#include "MRCNCMachineSettings.h"
#include <array>

namespace MR
{

/// This singleton struct contains default settings for scene objects
/// \ingroup BasicStructuresGroup
class SceneSettings
{
public:

    enum Type
    {
        /// enable flat shading for all new mesh objects
        MeshFlatShading,
        /// on deserialization replace object properties with default values from SceneSettings and SceneColors
        UseDefaultScenePropertiesOnDeserialization,
        /// automatically detect flat shading on imported mesh objects
        DetectMeshFlatShading,
        /// total count
        Count
    };

    MRMESH_API static bool get( Type type );
    MRMESH_API static void set( Type type, bool value );

    MRMESH_API static const CNCMachineSettings& getCNCMachineSettings();
    MRMESH_API static void setCNCMachineSettings( const CNCMachineSettings& settings );
private:
    SceneSettings() = default;
    ~SceneSettings() = default;

    static SceneSettings& instance_();

    std::array<bool, size_t( Type::Count ) > settings_{ false, true, true };
    CNCMachineSettings cncMachineSettings_;
};

}
