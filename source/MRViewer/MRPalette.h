#pragma once

#include "MRViewerFwd.h"
#include "MRMesh/MRVector4.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRColor.h"
#include "MRPch/MRJson.h"
#include "MRMesh/MRExpected.h"
#include <imgui.h>
#include <filesystem>

namespace MR
{

/**
 * @brief Class to hold one dimension texture with value to UV mapping
 * 
 * Discrete mode will draw rectangle for each color
 * Continuous mode will draw interpolated color
 * 
 */
class Palette
{
public:
    // preset palette colors
    static inline const std::vector<Color> DefaultColors =
    {
        Color( Vector4f { 0.1f, 0.25f, 1.0f, 1.f } ),// almost blue   |
        Color( Vector4f { 0.15f, 0.5f, 0.75f,1.f } ),//               | -> "inside" the ref mesh
        Color( Vector4f { 0.2f, 0.75f, 0.5f, 1.f } ),//               |
        Color( Vector4f { 0.25f, 1.0f, 0.25f,1.f } ),// almost green -> zero distance
        Color( Vector4f { 0.5f, 0.75f, 0.2f, 1.f } ),//               |
        Color( Vector4f { 0.75f, 0.5f, 0.15f,1.f } ),//               | -> "outside"
        Color( Vector4f { 1.0f, 0.25f, 0.1f, 1.f } ),// almost red    |
    };

    MRVIEWER_API Palette( const std::vector<Color>& colors );
    /**
     * @brief Set base palette colors
     * @detail colors.size() should be more or equal 2
     * for discrete palette using vector of colors calculated by mixing the base colors
     * i.e. base {blue, red} -> discrete 3 {blue, 0.5*blue + 0.5*red, red}
     */
    MRVIEWER_API void setBaseColors( const std::vector<Color>& colors );
    /**
     * @brief set range limits for palette (need for find color by value)
     * @detail all palette colors are evenly distributed between min and max
     */
    MRVIEWER_API void setRangeMinMax( float min, float max );
    /**
     * @brief set range limits for palette (need for find color by value)
     * @detail two half palette colors are evenly distributed between MinNeg / MaxNeg and MinPos / MaxPos
     * for values between MaxNeg / MinPos return one color (from center palette)
     */
    MRVIEWER_API void setRangeMinMaxNegPos( float minNeg, float maxNeg, float minPos, float maxPos );

    // set number of different colors for discrete palette
    MRVIEWER_API void setDiscretizationNumber( int discretization );
    // set palette type (linear / discrete)
    MRVIEWER_API void setFilterType( FilterType type );

    // Discrete: bar consists of single colored rectangles for each initial color
    // Linear (default): color is changing from one to another during initial color list
    MRVIEWER_API void draw( const ImVec2& pose, const ImVec2& size );

    // structure for label
    struct MRVIEWER_CLASS Label
    {
        float value = 0.f; // label position according normal scale (from Min to Max)
        std::string text; // label text

        Label() = default;
        MRVIEWER_API Label( float val, std::string text );
    };
    // reset labels to standard view
    MRVIEWER_API void resetLabels();
    // set labels manually
    MRVIEWER_API void setCustomLabels( const std::vector<Label>& labels );
    // set labels visible
    MRVIEWER_API void setLabelsVisible( bool visible );


    // Setup palette data from JsonValue
    // @return return true if loading successful
    MRVIEWER_API bool loadFromJson( const Json::Value& root );
    // Serialize this palette data to JsonValue
    MRVIEWER_API void saveCurrentToJson( Json::Value& root ) const;


    // return color according val and setting filter type
    // Discrete: returns color for given value
    // Linear: get interpolated color from the init vector
    MRVIEWER_API Color getColor( float val );

    const MeshTexture& getTexture() const
    {
        return texture_;
    };

    // get UV coordinate in palette for given value
    MRVIEWER_API UVCoord getUVcoord( float val );

    // base parameters of palette
    struct Parameters
    {
        std::vector<float> ranges = { 0.f, 1.f }; // range limits for palette
        std::vector<Color> baseColors; // palette base colors (need for calculate real colors according discretization)
        int discretization = 7; // number of different colors for discrete palette
    };

    MRVIEWER_API const Parameters& getParameters() const;

    // returns formated string for this value of palette
    MRVIEWER_API std::string getStringValue( float value );
    // returns maximal label count
    MRVIEWER_API int getMaxLabelCount();
    // sets maximal label count
    MRVIEWER_API void setMaxLabelCount( int val );
private:
    void setRangeLimits_( const std::vector<float>& ranges );

    void updateDiscretizatedColors_();
    Color getBaseColor_( float val );


    // fill labels with equal distance between
    void setUniformLabels_();
    // first label is equal to min value, last - to the max val
    // don't use with MinMaxNegPos mode
    void setZeroCentredLabels_();

    void updateCustomLabels_();

    void sortLabels_();

    std::vector<Label> customLabels_;
    std::vector<Label> labels_;
    bool showLabels_ = false;

    // stores OpenGL textures. Change useDiscrete_ to switch between them
    MeshTexture texture_;

    Parameters parameters_;

    bool isWindowOpen_ = false;

    bool useCustomLabels_ = false;

    int maxLabelCount_ = 0;

    static void resizeCallback_( ImGuiSizeCallbackData* data );
};

/// Class to save and load user palette presets
class PalettePresets
{
public:
    /// gets names of existing presets
    MRVIEWER_API static const std::vector<std::string>& getPresetNames();
    /// loads existing preset to given palette \n
    /// returns true if load was succeed
    MRVIEWER_API static bool loadPreset( const std::string& name, Palette& palette );
    /// saves given palette to preset with given name
    MRVIEWER_API static VoidOrErrStr savePreset( const std::string& name, const Palette& palette );
    /// returns path to presets folder
    MRVIEWER_API static std::filesystem::path getPalettePresetsFolder();
private:
    PalettePresets();
    ~PalettePresets() = default;

    std::vector<std::string> names_;
    
    void update_();

    static PalettePresets& instance_();
};

}
