#pragma once
#include "exports.h"
#include "MRMesh/MRColor.h"
#include "MRPch/MRJson.h"
#include <filesystem>
#include <vector>
#include <array>
#include <optional>

namespace MR
{

// This singleton is born to manage color-themes loading and saving, and apply it to app
// It can handle scene colors (default mesh colors, etc.) and Dear ImGui colors
class ColorTheme
{
public:
    enum class Preset
    {
        Dark,
        Light,
        Count
    };
    MRVIEWER_API static Preset getPreset();
    MRVIEWER_API static const char* getPresetName( Preset type );


    enum class Type
    {
        Default,
        User
    };

    MRVIEWER_API static Type getThemeType();
    MRVIEWER_API static const std::string& getThemeName();

    // Setup this struct
    MRVIEWER_API static void setupByTypeName( Type type, const std::string& name );
    MRVIEWER_API static void setupDefaultDark();
    MRVIEWER_API static void setupDefaultLight();
    MRVIEWER_API static void setupUserTheme( const std::string& themeName );
    // Setup this struct from serialized color-theme file
    MRVIEWER_API static void setupFromFile( const std::filesystem::path& path );
    // Setup this struct from Json value
    MRVIEWER_API static void setupFromJson( const Json::Value& value );

    // Setup this struct from current application colors, and serialize them to file
    // gets scene colors from first ObjectMesh, if it is present
    // gets active viewport background color
    MRVIEWER_API static void serializeCurrentToFile( const std::filesystem::path& path );
    // Setup this struct from current application colors, and serialize them to jsonValue
    // gets scene colors from first ObjectMesh, if it is present
    // gets active viewport background color
    MRVIEWER_API static void serializeCurrentToJson( Json::Value& root );

    // Applies colors stored in this struct to application
    // really some colors of this theme are applied deferred on next frame because of ImGui::PushStyleColor problem
    // note that struct should be initialized when apply is called
    // initialized in this scope means that structure has it's own values for colors
    MRVIEWER_API static void apply();
    // True if this structure is filled with colors, false if empty
    MRVIEWER_API static bool isInitialized();

    // Color types used in ribbon
    enum class RibbonColorsType
    {
        Background,
        HeaderBackground,
        HeaderSeparator,
        TopPanelBackground,

        QuickAccessBackground,

        Borders,

        TabHovered,
        TabClicked,

        TabActive,
        TabActiveHovered,
        TabActiveClicked,

        TabText,
        TabActiveText,

        ToolbarHovered,
        ToolbarClicked,

        Text,
        TextEnabled,
        TextDisabled,

        RibbonButtonHovered,
        RibbonButtonClicked,
        RibbonButtonActive,
        RibbonButtonActiveHovered,
        RibbonButtonActiveClicked,

        ThirdpartyBackground,

        FrameBackground,
        CollapseHeaderBackground,

        SelectedObjectText,
        SelectedObjectFrame,

        GradientStart,
        GradientEnd,

        Count
    };
    // Getter and setter for ribbon colors
    MRVIEWER_API static void setRibbonColor( const Color& color, RibbonColorsType type );
    MRVIEWER_API static const Color& getRibbonColor( RibbonColorsType type );
    MRVIEWER_API static const char* getRibbonColorTypeName( RibbonColorsType type );

    enum class ViewportColorsType
    {
        Background,
        Borders,
        Count
    };

    // Getter and setter for viewport colors
    MRVIEWER_API static void setViewportColor( const Color& color, ViewportColorsType type );
    MRVIEWER_API static const Color& getViewportColor( ViewportColorsType type );
    MRVIEWER_API static const char* getViewportColorTypeName( ViewportColorsType type );

    // Returns directory where user's custom themes are stored
    MRVIEWER_API static std::filesystem::path getUserThemesDirectory();

    // Reset ImGui style sizes and colors
    MRVIEWER_API static void resetImGuiStyle();
private:
    ColorTheme() = default;
    ~ColorTheme() = default;

    static ColorTheme& instance_();

    std::vector<Color> sceneColors_;
    Preset themePreset_ = Preset::Dark;
    std::array<Color, size_t( RibbonColorsType::Count )> newUIColors_;
    std::array<Color, size_t( ViewportColorsType::Count )> viewportColors_;

    Type type_{ Type::Default };
    std::string themeName_;
};

}