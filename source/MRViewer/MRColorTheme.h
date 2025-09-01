#pragma once
#include "exports.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRSignal.h"
#include <json/forwards.h>
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
    MRVIEWER_API static ColorTheme& instance();

    enum class Preset
    {
        Dark,
        Default = Dark,
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
    MRVIEWER_API static void setupFromFile( const std::filesystem::path& path, Type type = Type::User );
    // Setup this struct from Json value
    MRVIEWER_API static void setupFromJson( const Json::Value& value, Type type = Type::User );

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
        BackgroundSecStyle,
        HeaderBackground,
        HeaderSeparator,
        TopPanelBackground,
        TopPanelSearchBackground,

        QuickAccessBackground,

        Borders,

        TabHovered,
        TabClicked,
        TabActive,
        TabActiveHovered,
        TabActiveClicked,
        TabText,
        TabActiveText,

        DialogTab,
        DialogTabHovered,
        DialogTabActive,
        DialogTabActiveHovered,
        DialogTabText,
        DialogTabActiveText,

        ToolbarHovered,
        ToolbarClicked,

        ModalBackground,

        Text,
        TextEnabled,
        TextDisabled,
        TextSelectedBg,

        RibbonButtonHovered,
        RibbonButtonClicked,
        RibbonButtonActive,
        RibbonButtonActiveHovered,
        RibbonButtonActiveClicked,

        ThirdpartyBackground,
        ProgressBarBackground,
        FrameBackground,
        CollapseHeaderBackground,

        SelectedObjectText,
        SelectedObjectFrame,

        GradientStart,
        GradientEnd,

        TextContrastBackground,

        GradBtnStart,
        GradBtnHoverStart,
        GradBtnActiveStart,
        GradBtnDisableStart,
        GradBtnEnd,
        GradBtnHoverEnd,
        GradBtnActiveEnd,
        GradBtnDisableEnd,
        GradBtnText,

        GradBtnSecStyleStart,
        GradBtnSecStyleHoverStart,
        GradBtnSecStyleActiveStart,
        GradBtnSecStyleEnd,
        GradBtnSecStyleHoverEnd,
        GradBtnSecStyleActiveEnd,

        Grid,

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
        AxisX,
        AxisY,
        AxisZ,
        Count
    };

    // Getter and setter for viewport colors
    MRVIEWER_API static void setViewportColor( const Color& color, ViewportColorsType type );
    MRVIEWER_API static const Color& getViewportColor( ViewportColorsType type );
    MRVIEWER_API static const char* getViewportColorTypeName( ViewportColorsType type );

    // Returns directory where user's custom themes are stored
    MRVIEWER_API static std::filesystem::path getUserThemesDirectory();

    // Find available custom themes
    MRVIEWER_API static void updateUserThemesList();
    // Return list of found custom themes
    MRVIEWER_API static std::vector<std::string> foundUserThemes();

    // Reset ImGui style sizes and colors, and apply menu scaling to it
    MRVIEWER_API static void resetImGuiStyle();

    /// Connects given slot to receive signal on every Color Theme change, triggered in apply
    MRVIEWER_API static boost::signals2::connection onChanged( const std::function<void()>& slot, boost::signals2::connect_position position = boost::signals2::at_back );

private:
    ColorTheme() = default;
    ~ColorTheme() = default;


    std::vector<Color> sceneColors_;
    Preset themePreset_ = Preset::Dark;
    std::array<Color, size_t( RibbonColorsType::Count )> newUIColors_;
    std::array<Color, size_t( ViewportColorsType::Count )> viewportColors_;

    Type type_{ Type::Default };
    std::string themeName_;

    std::vector<std::string> foundUserThemes_;

    using ChangedSignal = Signal<void()>;
    ChangedSignal changedSignal_;
};

}
