#pragma once
#include "MRMesh/MRFinally.h"
#include "MRPch/MRFmt.h"
#include "MRNotificationType.h"
#include "exports.h"
#include "MRUnits.h"
#include "MRVectorTraits.h"
#include "MRImGui.h"
#include "MRColorTheme.h"
#include <span>
#include <string>
#include <optional>

namespace MR
{

class ImGuiImage;

namespace UI
{

// enumeration texture types
enum class TextureType
{
    Mono,
    Gradient,
    GradientBtn,
    GradientBtnSecond,
    GradientBtnGray,
    RainbowRect,
    Count
};

// get texture by type
MRVIEWER_API std::unique_ptr<ImGuiImage>& getTexture( TextureType type );

/// init internal parameters
MRVIEWER_API void init();

/// parameters to customize buttonEx
struct ButtonCustomizationParams
{
    /// If false, the button is grayed out and can't be clicked.
    bool enabled = true;

    /// imgui flags for this button
    ImGuiButtonFlags flags = ImGuiButtonFlags_None;

    /// gradient texture other than default
    /// {start, hover_start, acitve_start, disabled_start,
    ///  end, hover_end, acitve_end, disabled_end  }
    ImGuiImage* customTexture = nullptr;

    /// force use imgui background if !customTexture
    bool forceImGuiBackground = false;

    /// force use if ImGuiCol_Text for text
    bool forceImguiTextColor = false;
    /// show border or not
    bool border = false;

    /// draw line under first letter of label
    bool underlineFirstLetter = false;

    /// Allow interacting with this button from `UI::TestEngine`.
    bool enableTestEngine = true;

    /// if not empty, force use this string as name in TestEngine
    std::string testEngineName;
};

struct ButtonIconCustomizationParams
{
    // basic customization parameters
    ButtonCustomizationParams baseParams;

    // if false - text is to the right
    bool textUnderImage = true;
};

struct PlotAxis
{
    // the point from which the axes will be drawn
    ImVec2 startAxisPoint;

    // size plot by axis
    float size;
    // optimal length between dashes
    float optimalLenth = 10.0f;
    // the minimum value of the axis
    float minValue = 0.0f;
    // the maximal value of the axis
    float maxValue = 1.0f;
    // sign every nth dash
    size_t textDashIndicesStep = 1;

    // length dash without text
    float lenDash = 8.0f;
    // length dash with text
    float lenDashWithText = 12.0f;
    // text offset from dash
    float textPadding = 3.0f;
    // the format of the text for labels
    VarUnitToStringParams labelFormatParams;
};

/// returns true if button is pressed in this frame, preserve its further processing in viewer keyboard events system if taken here
MRVIEWER_API bool checkKey( ImGuiKey passedKey );

/// draw gradient button, which can be disabled (active = false)
[[deprecated( "Use UI::buttonEx( label, size, params ) instead" )]]
MRVIEWER_API bool buttonEx( const char* label,bool active, const Vector2f& size = Vector2f( 0, 0 ),
    ImGuiButtonFlags flags = ImGuiButtonFlags_None, const ButtonCustomizationParams& customParams = {} );

/// draw gradient button, which can be customized
MRVIEWER_API bool buttonEx( const char* label, const Vector2f& size = Vector2f( 0, 0 ), const ButtonCustomizationParams& customParams = {} );
/// draw gradient button, which can be disabled (active = false)
/// returns true if button is clicked in this frame, or key is pressed (optional)
MRVIEWER_API bool button( const char* label, bool active, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None );
/// draw gradient button
/// returns true if button is clicked in this frame, or key is pressed (optional)
inline bool button( const char* label, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None )
{
    return button( label, true, size, key );
}
/// draw gradient button with the ordinary button size
/// returns true if button is clicked in this frame, or key is pressed (optional)
MRVIEWER_API bool buttonCommonSize( const char* label, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None );
/// draw button with same logic as radioButton
MRVIEWER_API bool buttonUnique( const char* label, int* value, int ownValue, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None );

// draw dash with text along the horizontal axis
MRVIEWER_API void drawPoltHorizontalAxis( float menuScaling, const PlotAxis& plotAxis );
// draw dash with text along the vertical axis
MRVIEWER_API void drawPoltVerticalAxis( float menuScaling, const PlotAxis& plotAxis );

// draw a button with an icon and text under it
MRVIEWER_API bool buttonIconEx(
    const std::string& name,
    const Vector2f& iconSize,
    const std::string& text,
    const ImVec2& buttonSize,
    const ButtonIconCustomizationParams& params = {} );

// button with a gradient, always active
inline bool buttonIcon( const std::string& name, const Vector2f& iconSize, const std::string& text, const ImVec2& buttonSize )
{
    return buttonIconEx( name, iconSize, text, buttonSize );
}
// button without a gradient, always active, configurable by an external style
inline bool buttonIconFlatBG(
    const std::string& name,
    const Vector2f& iconSize,
    const std::string& text,
    const ImVec2& buttonSize,
    bool textUnderIcon = true,
    ImGuiKey key = ImGuiKey_None )
{
    ButtonIconCustomizationParams params;
    params.baseParams.forceImGuiBackground = true;
    params.baseParams.forceImguiTextColor = true;
    params.textUnderImage = textUnderIcon;
    params.baseParams.underlineFirstLetter = std::string_view( ImGui::GetKeyName( key ) ) == std::string_view( text.c_str(), 1 );
    return buttonIconEx( name, iconSize, text, buttonSize, params ) || checkKey( key );
}
/// draw button with icon same logic as radioButton
/// the colors of the internal style are used
MRVIEWER_API bool buttonUniqueIcon(
    const std::string& iconName,
    const Vector2f& iconSize,
    const std::string& text,
    const ImVec2& buttonSize,
    int* value,
    int ownValue,
    bool textUnderIcon = true,
    ImGuiKey key = ImGuiKey_None );


/// draws checkbox-like toggle (enabled/disabled states)(O=)/(=O)
MRVIEWER_API bool toggle( const char* label, bool* value );
/// draw gradient checkbox
MRVIEWER_API bool checkbox( const char* label, bool* value );
/// If `valueOverride` is specified, then the checkbox is disabled and that value is displayed instead of `value`.
MRVIEWER_API bool checkboxOrFixedValue( const char* label, bool* value, std::optional<bool> valueOverride );
/// If valid is false checkbox is disabled. Same as `checkboxOrFixedValue( ..., valid ? nullopt : false )`.
MRVIEWER_API bool checkboxValid( const char* label, bool* value, bool valid );
/// draw gradient checkbox with mixed state
MRVIEWER_API bool checkboxMixed( const char* label, bool* value, bool mixed );
/// draw gradient checkbox
template <typename Getter, typename Setter>
bool checkbox( const char* label, Getter get, Setter set )
{
    bool value = get();
    bool ret = checkbox( label, &value );
    set( value );
    return ret;
}
template <typename Getter, typename Setter>
bool checkboxValid( const char* label, Getter get, Setter set, bool valid )
{
    bool value = get();
    bool ret = checkboxValid( label, &value, valid );
    set( value );
    return ret;
}

/// Draw a checkbox toggling one or more bits in the mask.
template <typename T>
bool checkboxFlags( const char* label, T& target, T flags )
{
    bool value = bool( target & flags );
    bool mixed = value && ( target & flags ) != flags;
    if ( checkboxMixed( label, &value, mixed ) )
    {
        if ( value )
            target |= flags;
        else
            target &= ~flags;
        return true;
    }
    return false;
}

struct CheckboxOrModifierState
{
    // The persistent value of this setting, as set by the user by clicking the checkbox.
    bool baseValue = false;
    // Whether the setting is currently inverted because the modifier is held.
    bool modifierHeld = false;

    // You usually want to read this instead of the variables above.
    // Returns `baseValue`, but inverted if `modifierHeld` is set.
    [[nodiscard]] explicit operator bool() const { return baseValue != modifierHeld; }
};

/// Draws a checkbox, that gets inverted while a modifier key is held. Use `value`'s `operator bool` to read the final value. (E.g. `if (value)`, and so on.)
/// `modifiers` must be one or more of `ImGuiMod_{Shift,Alt,Ctrl}`.
/// By default ignores all modifiers not in `modifiers`. But if `respectedModifiers` is specified, then only ignores modifiers not included in it.
/// `respectedModifiers` must be a superset of `modifiers`. `-1` has special meaning, making it same as `modifiers` (which makes it have no effect).
/// In other words, modifiers are checked like this:
///     mask = respectedModifiers != -1 ? respectedModifiers : modifiers;
///     modifiersHeld = (ImGui::GetIO().KeyMods & mask) == modifiers;
/// If `valueOverride` is specified, then acts as `checkboxOrFixedValue`: disables the checkbox and displays that value instead of the real one,
///   and also pretends that the modifier isn't held.
MRVIEWER_API bool checkboxOrModifier( const char* label, CheckboxOrModifierState& value, int modifiers, int respectedModifiers = -1, std::optional<bool> valueOverride = {} );


/// draw gradient radio button
MRVIEWER_API bool radioButton( const char* label, int* value, int valButton );
/// If `valueOverride` is specified, then the radio button is disabled and that value is displayed instead of `value`.
MRVIEWER_API bool radioButtonOrFixedValue( const char* label, int* value, int valButton, std::optional<int> valueOverride );

struct RadioButtonOrModifierState
{
    // The permanent value of this setting, as set by the user by clicking the radio button.
    int value{};
    // The value that is displayed, and to be used - can differ from `value` if modifiers are pressed.
    int effectiveValue{};

    // The effective value, affected by modifiers.
    [[nodiscard]] explicit operator int() const
    {
        return effectiveValue;
    }
};

/// Draws a radio button, that can be affected by modifier keys.
/// `modifiers` must be one or more of `ImGuiMod_{Shift,Alt,Ctrl}`.
/// By default ignores all modifiers not in `modifiers`. But if `respectedModifiers` is specified, then only ignores modifiers not included in it.
/// `respectedModifiers` must be a superset of `modifiers`. `-1` has special meaning, making it same as `modifiers` (which makes it have no effect).
/// If `valueOverride` is specified, then acts as `radioButtonOrFixedValue`: disables the radio button and displays that value instead of the real one,
///   and also pretends that the modifier isn't held.
MRVIEWER_API bool radioButtonOrModifier( const char* label, RadioButtonOrModifierState& value, int valButton, int modifiers, int respectedModifiers = -1, std::optional<int> valueOverride = {} );

/// draw gradient color edit 4
MRVIEWER_API bool colorEdit4( const char* label, Vector4f& color, ImGuiColorEditFlags flags = ImGuiColorEditFlags_None );
MRVIEWER_API bool colorEdit4( const char* label, Color& color, ImGuiColorEditFlags flags = ImGuiColorEditFlags_None );

/// draw combo box
MRVIEWER_API bool combo( const char* label, int* v, const std::vector<std::string>& options,
    bool showPreview = true, const std::vector<std::string>& tooltips = {}, const std::string& defaultText = "Not selected" );

/// draw custom content combo box
MRVIEWER_API bool beginCombo( const char* label, const std::string& text = "Not selected", bool showPreview = true );
MRVIEWER_API void endCombo( bool showPreview = true );

/// Draws text input, should be used instead of `ImGui::InputText()`.
MRVIEWER_API bool inputText( const char* label, std::string& str, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = nullptr, void* user_data = nullptr );
/// This overload is for arrays, as opposed to `std::string`s.
MRVIEWER_API bool inputTextIntoArray( const char* label, char* array, std::size_t size, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = nullptr, void* user_data = nullptr );

/// Draws multiline text input, should be used instead of `ImGui::InputTextMultiline()`.
MRVIEWER_API bool inputTextMultiline( const char* label, std::string& str, const ImVec2& size = ImVec2(), ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = nullptr, void* user_data = nullptr );
/// This overload is for arrays, as opposed to `std::string`s.
MRVIEWER_API bool inputTextIntoArrayMultiline( const char* label, char* buf, size_t buf_size, const ImVec2& size = ImVec2(), ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = nullptr, void* user_data = nullptr );

struct CachedTextSize
{
    std::optional<ImVec2> cachedSize; // Reset this when manually modifying the text.
};
/// This version adds a horizontal scrollbar. Also it never draws the label, and uses full window width by default (so you can pass `0` as width).
MRVIEWER_API bool inputTextMultilineFullyScrollable( CachedTextSize& cache, const char* label, std::string& str, const ImVec2& size = ImVec2(), ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = nullptr, void* user_data = nullptr );
MRVIEWER_API bool inputTextIntoArrayMultilineFullyScrollable( CachedTextSize& cache, const char* label, char* buf, size_t buf_size, const ImVec2& size = ImVec2(), ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = nullptr, void* user_data = nullptr );

/// draw input text box with text aligned by center
MRVIEWER_API bool inputTextCentered( const char* label, std::string& str, float width = 0.0f, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = nullptr, void* user_data = nullptr );

/// draw read-only text box with text aligned by center
MRVIEWER_API void inputTextCenteredReadOnly( const char* label, const std::string& str, float width = 0.0f, const std::optional<ImVec4>& textColor = {}, const std::optional<ImVec4>& labelColor = {} );


namespace detail
{
    // A type-erased slider.
    MRVIEWER_API bool genericSlider( const char* label, ImGuiDataType data_type, void* p_data, const void* p_min, const void* p_max, const char* format, ImGuiSliderFlags flags );

    // Whether `T` is a scalar type that we can use with our widgets.
    template <typename T>
    concept Scalar = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

    // Whether `T` is a scalar or vector that we can use with our widgets.
    template <typename T>
    concept VectorOrScalar = Scalar<typename VectorTraits<T>::BaseType>;

    // Whether `Bound` is a valid min/max bound for `Target`.
    // That is, either the same type, or if `Target` is a vector, `Bound` can also be a scalar of the same type.
    template <typename Bound, typename Target>
    concept ValidBoundForTargetType =
        std::same_as<Bound, Target> ||
        ( VectorTraits<Bound>::size == 1 && std::same_as<typename VectorTraits<Bound>::BaseType, typename VectorTraits<Target>::BaseType> );

    // Whether `Speed` is a valid drag speed type for `Target`.
    // That is, either a single/vector of `float` or the same type as target (or its element if it's a vector).
    template <typename Speed, typename Target>
    concept ValidDragSpeedForTargetType =
        std::same_as<Speed, typename VectorTraits<Target>::BaseType> || std::same_as<Speed, float> ||
        std::same_as<Speed, Target> || std::same_as<Speed, typename VectorTraits<Target>::template ChangeBase<float>>;

    // A common code for sliders and other widgets dealing with measurement units.
    // `E` must be explicitly set to a measurement unit enum. The other template parameters are deduced.
    // `label` is the widget label, `v` is the target value.
    // `func` draws the widget for an individual scalar. We call it more than once for vectors.
    // `func` is `( const char* label, auto& elem, int i ) -> bool`.
    // It receives `elem` already converted to the display units (so you must convert min/max bounds manually). `i` is the element index for vectors.
    // When `v` is integral, `func` will be instantiated for both integral and floating-point element type. The latter is required if we're doing conversions.
    // NOTE: For integral `v`, in `func` you must look at the type of `elem` and convert your min/max bounds (etc) to the same type.
    // Notice `unitParams` being accepted by an lvalue reference. For convenience, we reset the `sourceUnit` in it before calling the user callback,
    //   since at that point no further conversions are necessary.
    template <UnitEnum E, VectorOrScalar T, typename F>
    [[nodiscard]] bool unitWidget( const char* label, T& v, UnitToStringParams<E>& unitParams, F&& func );

    // Some default slider parameters. For now they are hardcoded here, but we can move them elsewhere later.

    // Default drag speed for `UI::drag()`.
    template <UnitEnum E, VectorOrScalar T>
    requires ( VectorTraits<T>::size == 1 )
    [[nodiscard]] float getDefaultDragSpeed();

    // Default step speed for `UI::input()`.
    template <UnitEnum E, VectorOrScalar T, VectorOrScalar TargetType>
    [[nodiscard]] T getDefaultStep( bool fast );
}

// Default flags for `slider()` and `drag()` below.
inline constexpr int defaultSliderFlags = ImGuiSliderFlags_AlwaysClamp;

// Draw a slider.
// `E` must be specified explicitly, to one of: `NoUnit` `LengthUnit`, `AngleUnit`, ...
// By default, for angles `v` will be converted to degrees for display (but `vMin`, `vMax` are still in radians, same as `v`),
//   while length and unit-less values will be left as is. This can be customized in `unitParams` or globally (see `MRUnits.h`).
template <UnitEnum E, detail::VectorOrScalar T, detail::ValidBoundForTargetType<T> U = typename VectorTraits<T>::BaseType>
bool slider( const char* label, T& v, const U& vMin, const U& vMax, UnitToStringParams<E> unitParams = {}, ImGuiSliderFlags flags = defaultSliderFlags );

// Draw a dragging widget. Also includes [+] and [-] buttons (for integers only by default_, like `ImGui::Input`).
// `E` must be specified explicitly, to one of: `NoUnit` `LengthUnit`, `AngleUnit`, ...
// By default, for angles `v` will be converted to degrees for display (but `vSpeed` is still in radians, same as `v`),
//   while length and unit-less values will be left as is. This can be customized in `unitParams` or globally (see `MRUnits.h`).
// If only the min limit is specified, then the max limit is assumed to be the largest number.
template <UnitEnum E, detail::VectorOrScalar T, detail::ValidDragSpeedForTargetType<T> SpeedType = float, detail::ValidBoundForTargetType<T> U = typename VectorTraits<T>::BaseType>
bool drag( const char* label, T& v, SpeedType vSpeed = detail::getDefaultDragSpeed<E, SpeedType>(), const U& vMin = std::numeric_limits<U>::lowest(), const U& vMax = std::numeric_limits<U>::max(), UnitToStringParams<E> unitParams = {}, ImGuiSliderFlags flags = defaultSliderFlags, const U& step = detail::getDefaultStep<E, U, T>( false ), const U& stepFast = detail::getDefaultStep<E, U, T>( true ) );

// Like `drag()`, but clicking it immediately activates text input, so it's not actually draggable.
template <UnitEnum E, detail::VectorOrScalar T, detail::ValidBoundForTargetType<T> U = typename VectorTraits<T>::BaseType>
bool input( const char* label, T& v, const U& vMin = std::numeric_limits<U>::lowest(), const U& vMax = std::numeric_limits<U>::max(), UnitToStringParams<E> unitParams = {}, ImGuiSliderFlags flags = defaultSliderFlags );

// Draw a read-only copyable value.
// `E` must be specified explicitly, to one of: `NoUnit` `LengthUnit`, `AngleUnit`, ...
// By default, for angles `v` will be converted to degrees for display, while length and unit-less values will be left as is.
// This can be customized in `unitParams` or globally (see `MRUnits.h`).
template <UnitEnum E, detail::VectorOrScalar T>
void readOnlyValue( const char* label, const T& v, std::optional<ImVec4> textColor = {}, UnitToStringParams<E> unitParams = {}, std::optional<ImVec4> labelColor = {} );


/// returns icons font character for given notification type, and its color
MRVIEWER_API const std::pair<const char*, ImU32>& notificationChar( NotificationType type );

/// draws hint with corresponding mouse btn icon
MRVIEWER_API void mouseControlHint( ImGuiMouseButton btn, const std::string& hint, float scaling );

/// similar to ImGui::Text but use current text color with alpha channel = 0.5
MRVIEWER_API void transparentText( const char* fmt, ... );
/// similar to ImGui::TextWrapped but use current text color with alpha channel = 0.5
MRVIEWER_API void transparentTextWrapped( const char* fmt, ... );
/// similar to ImGui::TextWrapped but also have styled background and notification type indicator
MRVIEWER_API void notificationFrame( NotificationType type, const std::string& str, float scaling );

/// draw tooltip only if current item is hovered
MRVIEWER_API void setTooltipIfHovered( const std::string& text, float scaling );

/// Parameters for drawing custom separator
struct SeparatorParams
{
    /// optional icon in the left part of separator
    const ImGuiImage* icon{ nullptr };

    /// size of icon
    Vector2f iconSize; ///< scaling is applied inside `separator` function

    /// label at the left part of separator (drawn after icon if present)
    std::string label;

    /// framed text after label (might be used for some indications)
    std::string suffix;

    /// color of background frame behind suffix (if not present default ImGuiCol_FrameBg is used)
    std::optional<Color> suffixFrameColor;

    /// if set - use default spacing from ImGui::GetStyle()
    /// otherwise overrides it with ribbon constants
    bool forceImGuiSpacing = false;
};

/// separator line with customizations
MRVIEWER_API void separator( float scaling, const SeparatorParams& params );

/// add text with separator line
/// if issueCount is greater than zero, this number will be displayed in red color after the text.
/// If it equals zero - in green color
/// Otherwise it will not be displayed
MRVIEWER_API void separator( float scaling, const std::string& text = "", int issueCount = -1 );
MRVIEWER_API void separator(
    float scaling,
    const std::string& text,
    const ImVec4& color,
    const std::string& issueCount );
// separator line with icon and text
// iconSize icon size without scaling
MRVIEWER_API void separator( float scaling, const ImGuiImage& icon, const std::string& text, const Vector2f& iconSize = { 24.f, 24.f } );
MRVIEWER_API void separator( float scaling, const std::string& iconName, const std::string& text, const Vector2f& iconSize = { 24.f, 24.f } );

/// draws progress bar
/// note that even while scaling is given by argument size should still respect it
/// size: x(y)  < 0 - take all available width(height)
///       x(y) == 0 - use default width(height)
MRVIEWER_API void progressBar( float scaling, float fraction, const Vector2f& size = Vector2f( -1, 0 ) );

// create and append items into a TabBar: see corresponding ImGui:: functions
MRVIEWER_API bool beginTabBar( const char* str_id, ImGuiTabBarFlags flags = 0 );
MRVIEWER_API void endTabBar();
MRVIEWER_API bool beginTabItem( const char* label, bool* p_open = NULL, ImGuiTabItemFlags flags = 0 );
MRVIEWER_API void endTabItem();

/// Sets the vertical text position to match the text of a specific control on the same line
/// The control is assumed to contain standard one-line text with equal padding from top and bottom edges
/// The padding is control-specific
/// Sample usage:
///   UI::alignTextToFramePadding( ImGui::GetStyle().FramePadding.y );
///   // (with this value, it is equivalent to \ref ImGui::AlignTextToFramePadding() )
///   ImGui::Text( "Button:" );
///   ImGui::SameLine();
///   UI::buttonCommonSize( "Button" ); // the control we align to
MRVIEWER_API void alignTextToFramePadding( float padding );
/// Sets the vertical text position to match the text of a specific control on the same line
/// Same as \ref alignTextToFramePadding, but takes the full control height
/// Can be used, for example, for \ref UI::button with nondefault height
MRVIEWER_API void alignTextToControl( float controlHeight );
/// Specialization of \ref alignTextToFramePadding for \ref UI::radioButton
MRVIEWER_API void alignTextToRadioButton( float scaling );
/// Specialization of \ref alignTextToFramePadding for \ref UI::checkbox
MRVIEWER_API void alignTextToCheckBox( float scaling );
/// Specialization of \ref alignTextToFramePadding for \ref UI::button with default height
MRVIEWER_API void alignTextToButton( float scaling );

/// Select the background of the part of the current window from min to max.
/// If the min is not set, then the current position is taken.If max is not set, then the end of the window is taken.
/// Added some indentation if min or max is not set.
MRVIEWER_API void highlightWindowArea( float scaling, const ImVec2& min = {-1.0f, -1.0f}, const ImVec2& max = { -1.0f, -1.0f } );

// While this exists, it temporarily disables antialiasing for the lines drawn to this list.
class LineAntialiasingDisabler
{
    ImDrawList& list;
    ImDrawFlags oldFlags{};

public:
    LineAntialiasingDisabler( ImDrawList& list )
        : list( list ), oldFlags( list.Flags )
    {
        list.Flags &= ~ImDrawListFlags_AntiAliasedLines;
    }

    LineAntialiasingDisabler( const LineAntialiasingDisabler& ) = delete;
    LineAntialiasingDisabler& operator=( const LineAntialiasingDisabler& ) = delete;

    ~LineAntialiasingDisabler()
    {
        list.Flags = oldFlags;
    }
};

} // namespace UI

}

#include "MRUIStyle.ipp"
