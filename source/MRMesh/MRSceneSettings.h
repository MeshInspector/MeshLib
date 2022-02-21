#pragma once
#include "MRMeshFwd.h"
#include <array>

namespace MR
{

// This singleton struct contains default settings for scene objects
class SceneSettings
{
public:

    enum Type
    {
        MeshFlatShading,
        Count
    };

    MRMESH_API static bool get( Type type );
    MRMESH_API static void set( Type type, bool value );
private:
    SceneSettings() = default;
    ~SceneSettings() = default;

    static SceneSettings& instance_();

    std::array<bool, size_t( Type::Count ) > settings_{ false };
};

}
