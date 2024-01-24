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
        /// on deserialization replace object properties with default values from SceneSettings and SceneColors
        UseDefaultScenePropertiesOnDeserialization,
        /// total count
        Count
    };

    MRMESH_API static bool get( Type type );
    MRMESH_API static void set( Type type, bool value );

    /// Mesh faces shading mode
    enum class ShadingMode
    {
        AutoDetect,
        Smooth,
        Flat
    };

    /// Default shading mode for new mesh objects, or imported form files
    /// Tools may consider this setting when creating new meshes
    /// `AutoDetect`: choose depending of file format and mesh shape, fallback to smooth
    MRMESH_API static ShadingMode getDefaultShadingMode();
    MRMESH_API static void setDefaultShadingMode( ShadingMode mode );

    MRMESH_API static const CNCMachineSettings& getCNCMachineSettings();
    MRMESH_API static void setCNCMachineSettings( const CNCMachineSettings& settings );
private:
    SceneSettings() = default;
    ~SceneSettings() = default;

    static SceneSettings& instance_();

    std::array<bool, size_t( Type::Count ) > settings_{ true };
    ShadingMode defaultShadingMode_ = ShadingMode::AutoDetect;
    CNCMachineSettings cncMachineSettings_;
};

}
