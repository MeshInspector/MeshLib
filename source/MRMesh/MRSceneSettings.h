#pragma once
#include "MRMeshFwd.h"
#include "MRCNCMachineSettings.h"
#include <array>

namespace MR
{

/// Contains default settings for scene objects
/// \ingroup BasicStructuresGroup
namespace SceneSettings
{
    // Reset all scene settings to default values
    MRMESH_API void reset();

    enum class BoolType
    {
        /// on deserialization replace object properties with default values from SceneSettings and SceneColors
        UseDefaultScenePropertiesOnDeserialization,
        /// total count
        Count
    };

    enum class FloatType
    {
        FeaturePointsAlpha,
        FeatureLinesAlpha,
        FeatureMeshAlpha,
        FeatureSubPointsAlpha,
        FeatureSubLinesAlpha,
        FeatureSubMeshAlpha,
        // Line width of line features (line, circle, ...).
        FeatureLineWidth,
        // Line width of line subfeatures (axes, base circles, ...).
        FeatureSubLineWidth,
        // Size of the point feature.
        FeaturePointSize,
        // Size of point subfeatures (various centers).
        FeatureSubPointSize,
        // Ambient multiplication coefficient for ambientStrength for selected objects
        AmbientCoefSelectedObj,

        Count,
    };

    MRMESH_API bool get( BoolType type );
    MRMESH_API float get( FloatType type );
    MRMESH_API void set( BoolType type, bool value );
    MRMESH_API void set( FloatType type, float value );

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
    MRMESH_API ShadingMode getDefaultShadingMode();
    MRMESH_API void setDefaultShadingMode( ShadingMode mode );

    MRMESH_API const CNCMachineSettings& getCNCMachineSettings();
    MRMESH_API void setCNCMachineSettings( const CNCMachineSettings& settings );
}

}
