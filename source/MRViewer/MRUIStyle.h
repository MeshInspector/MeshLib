#pragma once
#include "MRMesh/MRFinally.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRUnits.h"
#include "MRViewer/MRVectorTraits.h"
#include "exports.h"
#include "imgui.h"
#include <span>
#include <string>
#include <optional>

namespace MR
{

class ImGuiImage;

namespace UI
{

/// init internal parameters
MRVIEWER_API void init();

/// parameters to customize buttonEx
struct ButtonCustomizationParams
{
    /// gradient texture other than default
    /// {start, hover_start, acitve_start, disabled_start,
    ///  end, hover_end, acitve_end, disabled_end  }
    ImGuiImage* customTexture = nullptr;
    /// force use if ImGuiCol_Text for text
    bool forceImguiTextColor = false;
    /// show border or not
    bool border = false;
};

/// draw gradient button, which can be disabled (active = false)
MRVIEWER_API bool buttonEx( const char* label, bool active, const Vector2f& size = Vector2f( 0, 0 ),
    ImGuiButtonFlags flags = ImGuiButtonFlags_None, const ButtonCustomizationParams& custmParams = {} );
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


/// draw gradient checkbox
MRVIEWER_API bool checkbox( const char* label, bool* value );
/// draw gradient checkbox
/// if valid is false checkbox is disabled
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


/// draw gradient radio button
MRVIEWER_API bool radioButton( const char* label, int* value, int valButton );

/// draw gradient color edit 4
MRVIEWER_API bool colorEdit4( const char* label, Vector4f& color, ImGuiColorEditFlags flags = ImGuiColorEditFlags_None );
MRVIEWER_API bool colorEdit4( const char* label, Color& color, ImGuiColorEditFlags flags = ImGuiColorEditFlags_None );

/// draw combo box
MRVIEWER_API bool combo( const char* label, int* v, const std::vector<std::string>& options,
    bool showPreview = true, const std::vector<std::string>& tooltips = {}, const std::string& defaultText = "Not selected" );

/// draw custom content combo box
MRVIEWER_API bool beginCombo( const char* label, const std::string& text = "Not selected", bool showPreview = true );
MRVIEWER_API void endCombo( bool showPreview = true );


MRVIEWER_API bool sliderFloat( const char* label, float* v, float v_min, float v_max, const char* format = "%.3f", ImGuiSliderFlags flags = 0 );
MRVIEWER_API bool sliderInt( const char* label, int* v, int v_min, int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0 );


/// draw input text box with text aligned by center
MRVIEWER_API bool inputTextCentered( const char* label, std::string& str, float width = 0.0f, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL );

/// draw read-only text box with text aligned by center
MRVIEWER_API void inputTextCenteredReadOnly( const char* label, const std::string& str, float width = 0.0f, const std::optional<ImVec4>& textColor = {} );


namespace detail
{
    // A type-erased slider.
    MRVIEWER_API bool genericSlider( const char* label, ImGuiDataType data_type, void* p_data, const void* p_min, const void* p_max, const char* format, ImGuiSliderFlags flags );

    // Whether `T` is a scalar type we can use with our widgets.
    template <typename T>
    concept ValidScalar = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

    // The `ImGuiDataType_??` constant for the given type.
    template <ValidScalar T>
    [[nodiscard]] constexpr int imGuiTypeEnum()
    {
        if constexpr ( std::is_same_v<T, float> )
            return ImGuiDataType_Float;
        else if constexpr ( std::is_same_v<T, double> )
            return ImGuiDataType_Double;
        if constexpr ( sizeof(T) == 1 )
            return std::is_signed_v<T> ? ImGuiDataType_S8 : ImGuiDataType_U8;
        else if constexpr ( sizeof(T) == 2 )
            return std::is_signed_v<T> ? ImGuiDataType_S16 : ImGuiDataType_U16;
        else if constexpr ( sizeof(T) == 4 )
            return std::is_signed_v<T> ? ImGuiDataType_S32 : ImGuiDataType_U32;
        else if constexpr ( sizeof(T) == 8 )
            return std::is_signed_v<T> ? ImGuiDataType_S64 : ImGuiDataType_U64;
        else
            static_assert( dependent_false<T>, "Unknown type." );
    }

    // Checks following:
    // `E` is a measurement unit enum (as per `MR::UnitEnum`), and
    // `T` is a scalar or a vector, with the base type (aka element type) satisfying `ValidScalar`.
    // If the base type is integral (not floating-point), `E` must be `NoUnit`.
    template <typename T, typename E>
    concept ValidTargetForUnitType =
        UnitEnum<E> &&
        ValidScalar<typename VectorTraits<T>::BaseType> &&
        ( ( !std::is_floating_point_v<typename VectorTraits<T>::BaseType> ) <=/*implies*/ std::is_same_v<E, NoUnit> );

    // Whether `Bound` is a valid min/max bound for `Target`.
    // That is, either the same type, or if `Target` is a vector, `Bound` can also be a scalar of the same type.
    template <typename Bound, typename Target>
    concept ValidBoundForTargetType =
        std::same_as<Bound, Target> ||
        ( VectorTraits<Bound>::size == 1 && std::same_as<typename VectorTraits<Bound>::BaseType, typename VectorTraits<Target>::BaseType> );

    // A common code for sliders and other widgets dealing with measurement units.
    // `E` must be explicitly set to a measurement unit enum. The other template parameters are deduced.
    // `label` is the widget label, `v` is the target value.
    // `func` draws the widget for an individual scalar. We call it more than once for vectors.
    // `func` is `( const char* label, A& elem, int i ) -> bool`.
    // It receives `elem` already converted to the display units (so you must convert min/max bounds manually). `i` is the element index for vectors.
    // Notice `unitParams` being accepted by an lvalue reference. For convenience, we reset the `sourceUnit` in it before calling the user callback,
    //   since at that point no further conversions are necessary.
    template <UnitEnum E, ValidTargetForUnitType<E> T, typename F>
    [[nodiscard]] bool unitWidget( const char* label, T& v, UnitToStringParams<E>& unitParams, F&& func )
    {
        constexpr bool conversionIsPossible = !std::is_integral_v<typename VectorTraits<T>::BaseType>;
        const bool needConversion = unitParams.sourceUnit && *unitParams.sourceUnit != unitParams.targetUnit;

        T* targetPtr = &v;
        T convertedValue{};

        const std::optional<E> originalSourceUnit = unitParams.sourceUnit;

        if constexpr ( conversionIsPossible )
        {
            if ( needConversion )
            {
                convertedValue = convertUnits( *unitParams.sourceUnit, unitParams.targetUnit, v );
                unitParams.sourceUnit.reset(); // To prevent the user from performing unnecessary conversions.
                targetPtr = &convertedValue;
            }
        }

        bool ret = false;
        if constexpr ( VectorTraits<T>::size == 1 )
        {
            ret = std::forward<F>( func )( label, *targetPtr, 0 );
        }
        else
        {
            ImGui::BeginGroup();
            MR_FINALLY{ ImGui::EndGroup(); };

            float width = ( ImGui::CalcItemWidth() - ImGui::GetStyle().ItemInnerSpacing.x * ( VectorTraits<T>::size - 1 ) ) / VectorTraits<T>::size;
            float prevX = 0;
            for ( int i = 0; i < VectorTraits<T>::size; i++ )
            {
                float x = std::round( width * ( i + 1 ) );
                if ( i > 0 )
                    ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
                ImGui::PushItemWidth( x - prevX );
                MR_FINALLY{ ImGui::PopItemWidth(); };
                prevX = x;
                bool elemChanged = func(
                    fmt::format( "{}{}##{}", i == VectorTraits<T>::size - 1 ? "" : "###", label, i ).c_str(),
                    VectorTraits<T>::getElem( i, *targetPtr ),
                    i
                );
                if ( elemChanged )
                {
                    ret = true;

                    if constexpr ( conversionIsPossible )
                    {
                        if ( needConversion )
                            VectorTraits<T>::getElem( i, v ) = convertUnits( unitParams.targetUnit, *originalSourceUnit, VectorTraits<T>::getElem( i, convertedValue ) );
                    }
                }
            }
        }

        return ret;
    }

    // All widgets created while this object is alive are read-only.
    struct ReadOnlyGuard
    {
        MRVIEWER_API ReadOnlyGuard();
        ReadOnlyGuard( const ReadOnlyGuard& ) = delete;
        ReadOnlyGuard& operator=( const ReadOnlyGuard& ) = delete;
        MRVIEWER_API ~ReadOnlyGuard();
    };

    // Some default slider parameters. For now they are hardcoded here, but we can move them elsewhere later.

    template <UnitEnum E, ValidTargetForUnitType<E> T>
    requires ( VectorTraits<T>::size == 1 )
    [[nodiscard]] T getDefaultDragSpeed()
    {
        if constexpr ( std::is_same_v<E, NoUnit> )
            return 1;
        else if constexpr ( std::is_same_v<E, LengthUnit> )
            return getDefaultUnitParams<LengthUnit>().targetUnit == LengthUnit::mm ? 1 : 1/16.f;
        else if constexpr ( std::is_same_v<E, AngleUnit> )
            return getDefaultUnitParams<AngleUnit>().targetUnit == AngleUnit::degrees ? 1 : PI_F / 128;
        else
            static_assert( dependent_false<E>, "Unknown measurement unit type." );
    }

    template <UnitEnum E, ValidTargetForUnitType<E> T>
    [[nodiscard]] T getDefaultStep( bool fast )
    {
        if constexpr ( std::is_integral_v<T> )
            return fast ? 100 : 1;
        else
            return 0;
    }


    template <UnitEnum E, ValidTargetForUnitType<E> T>
    std::string getDragRangeTooltip( T min, T max, const UnitToStringParams<E>& unitParams )
    {
        bool haveMin = min > std::numeric_limits<T>::lowest();
        bool haveMax = max < std::numeric_limits<T>::max();

        if constexpr ( !std::is_integral_v<typename VectorTraits<T>::BaseType> )
        {
            if ( unitParams.sourceUnit )
            {
                min = convertUnits( *unitParams.sourceUnit, unitParams.targetUnit, min );
                max = convertUnits( *unitParams.sourceUnit, unitParams.targetUnit, max );
            }
        }

        if ( haveMin && haveMax )
            return fmt::format( "Range: {} .. {}", min, max );
        if ( haveMin )
            return fmt::format( "Range: at least {}", min );
        if ( haveMax )
            return fmt::format( "Range: at most {}", max );
        return "";
    }

    MRVIEWER_API void drawDragTooltip( std::string rangeText );
}

// Default flags for `slider()` and `drag()` below.
inline constexpr int defaultSliderFlags = ImGuiSliderFlags_AlwaysClamp;

// Draw a slider.
// `E` must be specified explicitly, to one of: `NoUnit` `LengthUnit`, `AngleUnit`.
// By default, for angles `v` will be converted to degrees for display (but `vMin`, `vMax` are still in radians, same as `v`),
//   while length and unit-less values will be left as is. This can be customized in `unitParams` or globally (see `MRUnits.h`).
template <UnitEnum E, detail::ValidTargetForUnitType<E> T, detail::ValidBoundForTargetType<T> U>
bool slider( const char* label, T& v, U vMin, U vMax, UnitToStringParams<E> unitParams = {}, ImGuiSliderFlags flags = defaultSliderFlags )
{
    // Adjust the parameters:
    // Don't strip trailing zeroes, otherwise the numbers jump too much.
    unitParams.stripTrailingZeroes = false;

    if constexpr ( !std::is_integral_v<typename VectorTraits<U>::BaseType> )
    {
        if ( unitParams.sourceUnit && *unitParams.sourceUnit != unitParams.targetUnit )
        {
            // Without this flag the value changes slightly when you release the mouse.
            flags |= ImGuiSliderFlags_NoRoundToFormat;
            vMin = convertUnits( *unitParams.sourceUnit, unitParams.targetUnit, vMin );
            vMax = convertUnits( *unitParams.sourceUnit, unitParams.targetUnit, vMax );
        }
    }

    return detail::unitWidget( label, v, unitParams,
        [&]<typename ElemType>( const char* elemLabel, ElemType& elemVal, int i )
        {
            return detail::genericSlider(
                elemLabel, detail::imGuiTypeEnum<ElemType>(), &elemVal,
                &VectorTraits<U>::getElem( i, vMin ),
                &VectorTraits<U>::getElem( i, vMax ),
                valueToString<E>( elemVal, unitParams ).c_str(), flags
            );
        } );
}

// Draw a dragging widget.
// `E` must be specified explicitly, to one of: `NoUnit` `LengthUnit`, `AngleUnit`.
// By default, for angles `v` will be converted to degrees for display (but `vSpeed` is still in radians, same as `v`),
//   while length and unit-less values will be left as is. This can be customized in `unitParams` or globally (see `MRUnits.h`).
template <UnitEnum E, detail::ValidTargetForUnitType<E> T, detail::ValidBoundForTargetType<T> U = typename VectorTraits<T>::BaseType>
bool drag( const char* label, T& v, U vSpeed = detail::getDefaultDragSpeed<E, U>(), U vMin = 0, U vMax = 0, UnitToStringParams<E> unitParams = {}, ImGuiSliderFlags flags = defaultSliderFlags )
{
    // Adjust the parameters:
    // Don't strip trailing zeroes, otherwise the numbers jump too much.
    unitParams.stripTrailingZeroes = false;

    if constexpr ( !std::is_integral_v<typename VectorTraits<U>::BaseType> )
    {
        if ( unitParams.sourceUnit && *unitParams.sourceUnit != unitParams.targetUnit )
        {
            // Without this flag the value changes slightly when you release the mouse.
            flags |= ImGuiSliderFlags_NoRoundToFormat;
            vSpeed = convertUnits( *unitParams.sourceUnit, unitParams.targetUnit, vSpeed );
            vMin = convertUnits( *unitParams.sourceUnit, unitParams.targetUnit, vMin );
            vMax = convertUnits( *unitParams.sourceUnit, unitParams.targetUnit, vMax );
        }
    }

    return detail::unitWidget( label, v, unitParams,
        [&]<typename ElemType>( const char* elemLabel, ElemType& elemVal, int i )
        {
            bool ret = ImGui::DragScalar(
                elemLabel, detail::imGuiTypeEnum<ElemType>(), &elemVal,
                VectorTraits<U>::getElem( i, vSpeed ),
                &VectorTraits<U>::getElem( i, vMin ),
                &VectorTraits<U>::getElem( i, vMax ),
                valueToString<E>( elemVal, unitParams ).c_str(), flags
            );
            detail::drawDragTooltip( detail::getDragRangeTooltip( vMin, vMax, unitParams ) );
            return ret;
        } );
}

// Draw a textbox.
// `E` must be specified explicitly, to one of: `NoUnit` `LengthUnit`, `AngleUnit`.
// By default, for angles `v` will be converted to degrees for display, while length and unit-less values will be left as is.
// This can be customized in `unitParams` or globally (see `MRUnits.h`).
template <UnitEnum E, detail::ValidTargetForUnitType<E> T, detail::ValidBoundForTargetType<T> U = typename VectorTraits<T>::BaseType>
bool input(
    const char* label, T& v,
    // As with `ImGui::Drag...()`, steps default to zero for floating-point types.
    U vStep = detail::getDefaultStep<E, U>( false ),
    U vStepFast = detail::getDefaultStep<E, U>( true ),
    UnitToStringParams<E> unitParams = {},
    ImGuiInputTextFlags flags = 0
)
{
    if constexpr ( !std::is_integral_v<typename VectorTraits<U>::BaseType> )
    {
        if ( unitParams.sourceUnit && *unitParams.sourceUnit != unitParams.targetUnit )
        {
            vStep = convertUnits( *unitParams.sourceUnit, unitParams.targetUnit, vStep );
            vStepFast = convertUnits( *unitParams.sourceUnit, unitParams.targetUnit, vStepFast );
        }

    }
    return detail::unitWidget( label, v, unitParams,
        [&]<typename ElemType>( const char* elemLabel, ElemType& elemVal, int i )
        {
            (void)i;
            return ImGui::InputScalar( elemLabel, detail::imGuiTypeEnum<ElemType>(), &elemVal,
                &VectorTraits<U>::getElem( i, vStep ),
                &VectorTraits<U>::getElem( i, vStepFast ),
                valueToString<E>( elemVal, unitParams ).c_str(), flags
            );
        } );
}

// Draw a read-only copyable value.
// `E` must be specified explicitly, to one of: `NoUnit` `LengthUnit`, `AngleUnit`.
// By default, for angles `v` will be converted to degrees for display, while length and unit-less values will be left as is.
// This can be customized in `unitParams` or globally (see `MRUnits.h`).
template <UnitEnum E, detail::ValidTargetForUnitType<E> T>
void readOnlyValue( const char* label, const T& v, std::optional<ImVec4> textColor = {}, UnitToStringParams<E> unitParams = {} )
{
    (void)detail::unitWidget( label, const_cast<T&>( v ), unitParams,
        [&]( const char* elemLabel, auto& elemVal, int i )
        {
            (void)i;
            inputTextCenteredReadOnly( elemLabel, valueToString<E>( elemVal, unitParams ), ImGui::CalcItemWidth(), textColor );
            return false;
        } );
}


/// similar to ImGui::Text but use current text color with alpha channel = 0.5
MRVIEWER_API void transparentText( const char* fmt, ... );
/// similar to ImGui::TextWrapped but use current text color with alpha channel = 0.5
MRVIEWER_API void transparentTextWrapped( const char* fmt, ... );

/// draw tooltip only if current item is hovered
MRVIEWER_API void setTooltipIfHovered( const std::string& text, float scaling );

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

} // namespace UI

}
