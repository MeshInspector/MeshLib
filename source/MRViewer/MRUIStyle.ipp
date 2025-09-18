#pragma once

// A template implementation file for `MRUIStyle.h`. Include that file.

#include "MRUIStyle.h" // To help intellisense.
#include "MRUITestEngine.h"
#include "MRViewerInstance.h"
#include "MRMesh/MRString.h"

namespace MR::UI
{

namespace detail
{
    // `UI::drag()` uses this internally to draw tooltips.
    // Pass `getDragRangeTooltip()` as the parameter.
    MRVIEWER_API void drawDragTooltip( std::string rangeText );

    // Wraps `ImGui::MarkItemEdited()`, to avoid including `imgui_internal.h`.
    MRVIEWER_API void markItemEdited( ImGuiID id );

    // Checks if the active item ID matches the string. Wrapping it to avoid including `imgui_internal.h`.
    [[nodiscard]] MRVIEWER_API bool isItemActive( const char* name );


    // The `ImGuiDataType_??` constant for the given type.
    template <Scalar T>
    [[nodiscard]] constexpr int imGuiTypeEnum()
    {
        if constexpr ( std::is_same_v<T, float> )
            return ImGuiDataType_Float;
        else if constexpr ( std::is_same_v<T, double> )
            return ImGuiDataType_Double;
        else if constexpr ( sizeof(T) == 1 )
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

    template <UnitEnum E, VectorOrScalar T, typename F>
    [[nodiscard]] bool unitWidget( const char* label, T& v, UnitToStringParams<E>& unitParams, F&& func )
    {
        const std::optional<E> originalSourceUnit = unitParams.sourceUnit;

        // We need to jump through some hoops to handle integers.
        // If the input is an integer and if no conversion is needed, we draw the widget directly for that integer.
        // If the input is an integer and we do need a conversion, we draw a floating-point slider, then round back.

        const bool mustConvertUnits = !unitsAreEquivalent( unitParams.sourceUnit, unitParams.targetUnit );

        constexpr bool targetIsIntegral = std::is_integral_v<typename VectorTraits<T>::BaseType>;

        auto drawWidget = [&]<VectorOrScalar U>( U& value ) -> bool
        {
            auto onElemChanged = [&]( int i )
            {
                if constexpr ( !std::is_integral_v<typename VectorTraits<U>::BaseType> )
                {
                    // Convert back to source units.
                    if ( mustConvertUnits )
                        VectorTraits<T>::getElem( i, value ) = convertUnits( unitParams.targetUnit, originalSourceUnit, VectorTraits<T>::getElem( i, value ) );

                    if constexpr ( std::is_integral_v<typename VectorTraits<T>::BaseType> ) // aka `targetIsIntegral`, but MSVC 2019 chokes on that.
                    {
                        // Round back to integer, assigning to the original value.
                        VectorTraits<T>::getElem( i, v ) = (typename VectorTraits<T>::BaseType) std::round( VectorTraits<T>::getElem( i, value ) );
                    }
                    else if ( mustConvertUnits )
                    {
                        // Copy back to the unconverted original value.
                        VectorTraits<T>::getElem( i, v ) = VectorTraits<T>::getElem( i, value );
                    }
                }
                else
                {
                    (void)i;
                    // If `U` is integral, then it's guaranteed that the number is being modified inplace.
                }
            };

            bool ret = false;
            if constexpr ( VectorTraits<T>::size == 1 )
            {
                ret = std::forward<F>( func )( label, value, 0 );
                if ( ret )
                    onElemChanged( 0 );
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
                        VectorTraits<T>::getElem( i, value ),
                        i
                    );
                    if ( elemChanged )
                    {
                        ret = true;
                        onElemChanged( i );
                    }
                }
            }

            return ret;
        };

        if constexpr ( targetIsIntegral )
        {
            if ( !mustConvertUnits )
                return drawWidget( v );
        }

        // If `T` is an integer or a vector of interegers, changes its element type to `float`. Otherwise leaves it intact.
        using FloatingType = decltype( convertUnits( E{}, E{}, v ) );

        FloatingType valueCopy{};

        if ( mustConvertUnits || targetIsIntegral )
        {
            valueCopy = convertUnits( unitParams.sourceUnit, unitParams.targetUnit, v );
            unitParams.sourceUnit.reset(); // To prevent the user from performing unnecessary conversions.
            return drawWidget( valueCopy );
        }

        return drawWidget( v );
    }

    // See `drawDragTooltip()`.
    template <UnitEnum E, VectorOrScalar T>
    [[nodiscard]] std::string getDragRangeTooltip( T min, T max, const UnitToStringParams<E>& unitParams )
    {
        if ( !( min <= max ) )
            return "";

        bool haveMin = min > std::numeric_limits<T>::lowest();
        bool haveMax = max < std::numeric_limits<T>::max();

        if ( !haveMin && !haveMax )
            return "";

        std::string minString = valueToString<E>( min, unitParams );
        std::string maxString = valueToString<E>( max, unitParams );

        if ( haveMin && haveMax )
            return fmt::format( "Range: {} .. {}", minString, maxString );
        if ( haveMin )
            return fmt::format( "Range: at least {}", minString );
        if ( haveMax )
            return fmt::format( "Range: at most {}", maxString );
        return "";
    }

    [[nodiscard]] inline const char* getTestEngineLabelForVecElem( int i )
    {
        return std::array{ "x", "y", "z", "w" }[i];
    }
}

template <UnitEnum E, detail::VectorOrScalar T>
requires ( VectorTraits<T>::size == 1 )
[[nodiscard]] float getDefaultDragSpeed()
{
    if constexpr ( std::is_same_v<E, NoUnit> )
        return 1;
    else if constexpr ( std::is_same_v<E, LengthUnit> )
        return getDefaultUnitParams<LengthUnit>().targetUnit == LengthUnit::mm ? 1 : 1/16.f;
    else if constexpr ( std::is_same_v<E, AngleUnit> )
        return PI_F / 180;
    else
        return 1;
}

template <UnitEnum E, detail::VectorOrScalar T, detail::VectorOrScalar TargetType>
[[nodiscard]] T getDefaultStep( bool fast )
{
    if constexpr ( VectorTraits<TargetType>::size > 1 )
        return 0; // Multi-element widgets have no step buttons by default.
    else if constexpr ( std::is_integral_v<T> )
        return fast ? 100 : 1;
    else
        return 0;
}

template <UnitEnum E, detail::VectorOrScalar T, detail::ValidBoundForTargetType<T> U>
bool slider( const char* label, T& v, const U& vMin, const U& vMax, UnitToStringParams<E> unitParams, ImGuiSliderFlags flags )
{
    if constexpr ( !detail::Scalar<T> )
        UI::TestEngine::pushTree( label );
    MR_FINALLY{
        if constexpr ( !detail::Scalar<T> )
            UI::TestEngine::popTree();
    };

    auto fixedMin = convertUnits( unitParams.sourceUnit, unitParams.targetUnit, vMin );
    auto fixedMax = convertUnits( unitParams.sourceUnit, unitParams.targetUnit, vMax );

    // This flag is absolutely necessary when `if ( !unitsAreEquivalent( unitParams.sourceUnit, unitParams.targetUnit ) )`.
    //     Otherwise the value changes slightly when you release the mouse.
    // We also decided to add it unconditionally, to allow users to set fully custom values ignoring the rounding.
    //     It's slightly bad to not see the changes in the UI, but it's unclear how to fix that.
    flags |= ImGuiSliderFlags_NoRoundToFormat;

    // Guess the precision.
    if ( unitParams.style == NumberStyle::distributePrecision || unitParams.style == NumberStyle::normal )
        unitParams.precision = std::max( unitParams.precision, guessPrecision( fixedMin, fixedMax ) + int( unitParams.style == NumberStyle::distributePrecision ) );

    return detail::unitWidget( label, v, unitParams,
        [&]<typename ElemType>( const char* elemLabel, ElemType& elemVal, int i )
        {
            const ElemType *elemMin = nullptr;
            const ElemType *elemMax = nullptr;

            if constexpr ( std::is_integral_v<ElemType> )
            {
                elemMin = &VectorTraits<U>::getElem( i, vMin );
                elemMax = &VectorTraits<U>::getElem( i, vMax );
            }
            else
            {
                elemMin = &VectorTraits<decltype(fixedMin)>::getElem( i, fixedMin );
                elemMax = &VectorTraits<decltype(fixedMax)>::getElem( i, fixedMax );
            }

            if ( *elemMin <= *elemMax && bool( flags & ImGuiSliderFlags_AlwaysClamp ) ) // sometimes ImGui does not clamp it, so make sure that value is clamped
                elemVal = std::clamp( elemVal, *elemMin, *elemMax );

            // Don't strip trailing zeroes when active, otherwise the numbers jump too much.
            bool forceShowZeroes = unitParams.stripTrailingZeroes && isItemActive( elemLabel );
            if ( forceShowZeroes )
                unitParams.stripTrailingZeroes = false;
            MR_FINALLY{
                if ( forceShowZeroes )
                    unitParams.stripTrailingZeroes = true;
            };

            bool ret = detail::genericSlider(
                elemLabel, detail::imGuiTypeEnum<ElemType>(), &elemVal, elemMin, elemMax, valueToImGuiFormatString( elemVal, unitParams ).c_str(), flags
            );

            if ( ret ) // it is needed if we in drag mode to update frame with changed value after moving mouse
                incrementForceRedrawFrames();

            // Test engine stuff:
            if ( auto opt = TestEngine::createValue( detail::Scalar<T> ? label : detail::getTestEngineLabelForVecElem( i ), elemVal, *elemMin, *elemMax ) )
            {
                elemVal = *opt;
                ret = true;
                detail::markItemEdited( ImGui::GetItemID() );
            }

            return ret;
        } );
}

template <UnitEnum E, detail::VectorOrScalar T, detail::ValidDragSpeedForTargetType<T> SpeedType, detail::ValidBoundForTargetType<T> U>
bool drag( const char* label, T& v, SpeedType vSpeed, const U& vMin, const U& vMax, UnitToStringParams<E> unitParams, ImGuiSliderFlags flags, const U& step, const U& stepFast )
{
    if constexpr ( !detail::Scalar<T> )
        UI::TestEngine::pushTree( label );
    MR_FINALLY{
        if constexpr ( !detail::Scalar<T> )
            UI::TestEngine::popTree();
    };

    auto fixedSpeed = convertUnits( unitParams.sourceUnit, unitParams.targetUnit, vSpeed );
    auto fixedMin = convertUnits( unitParams.sourceUnit, unitParams.targetUnit, vMin );
    auto fixedMax = convertUnits( unitParams.sourceUnit, unitParams.targetUnit, vMax );
    auto fixedStep = convertUnits( unitParams.sourceUnit, unitParams.targetUnit, step );
    auto fixedStepFast = convertUnits( unitParams.sourceUnit, unitParams.targetUnit, stepFast );

    // This flag is absolutely necessary when `if ( !unitsAreEquivalent( unitParams.sourceUnit, unitParams.targetUnit ) )`.
    //     Otherwise the value changes slightly when you release the mouse.
    // We also decided to add it unconditionally, to allow users to set fully custom values ignoring the rounding.
    //     It's slightly bad to not see the changes in the UI, but it's unclear how to fix that.
    flags |= ImGuiSliderFlags_NoRoundToFormat;

    // Guess the precision.
    if ( unitParams.style == NumberStyle::distributePrecision || unitParams.style == NumberStyle::normal )
        unitParams.precision = std::max( unitParams.precision, guessPrecision( fixedSpeed ) + int( unitParams.style == NumberStyle::distributePrecision ) );

    return detail::unitWidget( label, v, unitParams,
        [&]<typename ElemType>( const char* elemLabel, ElemType& elemVal, int i )
        {
            const ElemType *elemMin = nullptr;
            const ElemType *elemMax = nullptr;
            const ElemType *elemStep = nullptr;
            const ElemType *elemStepFast = nullptr;

            if constexpr ( std::is_integral_v<ElemType> )
            {
                elemMin = &VectorTraits<U>::getElem( i, vMin );
                elemMax = &VectorTraits<U>::getElem( i, vMax );
                elemStep = &VectorTraits<U>::getElem( i, step );
                elemStepFast = &VectorTraits<U>::getElem( i, stepFast );
            }
            else
            {
                elemMin = &VectorTraits<decltype(fixedMin)>::getElem( i, fixedMin );
                elemMax = &VectorTraits<decltype(fixedMax)>::getElem( i, fixedMax );
                elemStep = &VectorTraits<decltype(fixedStep)>::getElem( i, fixedStep );
                elemStepFast = &VectorTraits<decltype(fixedStepFast)>::getElem( i, fixedStepFast );
            }

            if ( *elemMin <= *elemMax && bool( flags & ImGuiSliderFlags_AlwaysClamp ) ) // sometimes ImGui does not clamp it, so make sure that value is clamped
                elemVal = std::clamp( elemVal, *elemMin, *elemMax );

            bool plusMinusButtons = step > 0 && stepFast > 0;

            bool ret = false;

            // An arbitrary left offset for +/- buttons, to increase the separation to the text.
            // Even though it makes it wider than it should be in pixels, it looks more appealing this way, because of how the buttons are colored.
            float plusMinusButtonsLeftOffset = ImGui::GetStyle().FrameBorderSize;
            if ( plusMinusButtons )
            {
                ImGui::BeginGroup();
                // Here we make sure that the new width is not negative, because otherwise things break.
                // The min limit is arbitrary.
                float w = std::max( ImGui::CalcItemWidth() - ( ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x ) * 2 - plusMinusButtonsLeftOffset, ImGui::GetStyle().ItemSpacing.x );
                ImGui::PushItemWidth( w );
            }
            MR_FINALLY{
                if ( plusMinusButtons )
                {
                    ImGui::PopItemWidth();
                    ImGui::EndGroup();
                }
            };

            std::string elemLabelFixed = plusMinusButtons ? std::string( "###" ) + elemLabel : elemLabel;

            // Don't strip trailing zeroes when active, otherwise the numbers jump too much.
            bool forceShowZeroes = unitParams.stripTrailingZeroes && isItemActive( elemLabelFixed.c_str() );
            if ( forceShowZeroes )
                unitParams.stripTrailingZeroes = false;

            float dragY = ImGui::GetCursorPosY();
            ret = ImGui::DragScalar(
                elemLabelFixed.c_str(), detail::imGuiTypeEnum<ElemType>(), &elemVal,
                float( VectorTraits<SpeedType>::getElem( i, fixedSpeed ) ), elemMin, elemMax, valueToImGuiFormatString( elemVal, unitParams ).c_str(), flags
            );

            if ( ret )
            {
                // could be that after dragging value ImGui does not clamp value in first frame
                if ( *elemMin <= *elemMax && bool( flags & ImGuiSliderFlags_AlwaysClamp ) )
                    elemVal = std::clamp( elemVal, *elemMin, *elemMax );

                // it is needed if we in drag mode to update frame with changed value after moving mouse
                incrementForceRedrawFrames();
            }
            auto dragId = ImGui::GetItemID();

            if ( forceShowZeroes )
                unitParams.stripTrailingZeroes = true;

            detail::drawDragTooltip( detail::getDragRangeTooltip( VectorTraits<U>::getElem( i, fixedMin ), VectorTraits<U>::getElem( i, fixedMax ), unitParams ) );

            if ( plusMinusButtons )
            {
                ImGui::PushID( ( std::string( "PlusMinusButtons:" ) + elemLabel ).c_str() );
                // ImGui::PushButtonRepeat( true ); // Useless unless we can somehow avoid blocking the main thread while the button is pressed.
                MR_FINALLY{
                    // ImGui::PopButtonRepeat();
                    ImGui::PopID();
                };

                int action = 0;

                // U+2212 MINUS SIGN
                Vector2f buttonSize( ImGui::GetFrameHeight(), ImGui::GetFrameHeight() );
                ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
                ImGui::SetCursorPosY( dragY ); // Usually redundant, but when the user does something weird, this is sometimes required.
                action -= UI::buttonEx( "\xe2\x88\x92", buttonSize, { .enableTestEngine = false } );
                ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
                ImGui::SetCursorPosY( dragY );
                action += UI::buttonEx( "+", buttonSize, { .enableTestEngine = false } );

                if ( action )
                {
                    ret = true;
                    elemVal += ( ImGui::GetIO().KeyCtrl ? *elemStepFast : *elemStep ) * action;
                    if ( *elemMin <= *elemMax )
                        elemVal = std::clamp( elemVal, *elemMin, *elemMax );

                    detail::markItemEdited( dragId );
                }

                // The label.
                std::string_view labelView( label );
                auto pos = labelView.find( "##" );
                if ( pos > 0 )
                {
                    ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x + plusMinusButtonsLeftOffset );
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted( label, label + ( pos == std::string_view::npos ? labelView.size() : pos ) );
                }
            }

            // Test engine stuff:
            if ( auto opt = TestEngine::createValue( detail::Scalar<T> ? label : detail::getTestEngineLabelForVecElem( i ), elemVal, *elemMin, *elemMax ) )
            {
                elemVal = *opt;
                ret = true;
                detail::markItemEdited( ImGui::GetItemID() );
            }

            return ret;
        } );
}

template <UnitEnum E, detail::VectorOrScalar T, detail::ValidBoundForTargetType<T> U>
bool input( const char* label, T& v, const U& vMin, const U& vMax, UnitToStringParams<E> unitParams, ImGuiSliderFlags flags )
{
    // This is a hack to activate the input with a single click, by pretending that Ctrl is also held.
    if (
        ImGui::GetMousePos().x >= ImGui::GetCursorScreenPos().x &&
        ImGui::GetMousePos().y >= ImGui::GetCursorScreenPos().y &&
        ImGui::GetMousePos().x < ImGui::GetCursorScreenPos().x + ImGui::CalcItemWidth() &&
        ImGui::GetMousePos().y < ImGui::GetCursorScreenPos().y + ImGui::GetFrameHeight() &&
        ImGui::IsMouseClicked( ImGuiMouseButton_Left ) &&
        ImGui::GetIO().KeyMods == 0
    )
    {
        ImGui::GetIO().KeyCtrl = true;
    }

    return (drag<E>)( label, v, 0.f, vMin, vMax, std::move( unitParams ), flags );
}

template <UnitEnum E, detail::VectorOrScalar T>
void readOnlyValue( const char* label, const T& v, std::optional<ImVec4> textColor, UnitToStringParams<E> unitParams, std::optional<ImVec4> labelColor )
{
    (void)detail::unitWidget( label, const_cast<T&>( v ), unitParams,
        [&]( const char* elemLabel, auto& elemVal, int i )
        {
            (void)i;
            inputTextCenteredReadOnly( elemLabel, valueToString<E>( elemVal, unitParams ), ImGui::CalcItemWidth(), textColor, labelColor );
            return false;
        } );
}

template <UnitEnum E, detail::Scalar T, typename F>
bool plusMinusGeneric( const char* label, T& plus, T& minus, UnitToStringParams<E> unitToStringParams, F&& func )
{
    auto getStateKeyIsAsym = [&, ret = std::string{}]() mutable -> const std::string&
    {
        if ( ret.empty() )
            ret = fmt::format( "MR::plusMinus:{}", label );
        return ret;
    };

    auto getPlusName = [&, ret = std::string{}]() mutable -> const std::string&
    {
        if ( ret.empty() )
            ret = fmt::format( "###plus:{}", label );
        return ret;
    };
    auto getMinusName = [&, ret = std::string{}]() mutable -> const std::string&
    {
        if ( ret.empty() )
            ret = fmt::format( "###minus:{}", label );
        return ret;
    };

    // Display asymmetrically if...
    bool isAsym =
        // The values are different, or...
        plus != -minus ||
        // The state says so, or...
        StateStorage::readBool( getStateKeyIsAsym(), false ) ||
        // Any of the plus/minus widgets are active. To make sure we don't remove them while they're being edited.
        isItemActive( getPlusName().c_str() ) || isItemActive( getMinusName().c_str() )
    ;

    float toggleSymButtonWidth = ImGui::CalcTextSize( "+/- " ).x;
    float fullWidth = ImGui::CalcItemWidth();

    // Draw the symmetry toggle first (and the label). This is because we want any `IsItemActive()` checks after this widget to act on the plus/minus
    //   widgets (which we `BeginGroup()` together to get events at the same time).

    // The label.
    ImGui::GetWindowDrawList()->AddText(
        ImVec2( ImGui::GetCursorScreenPos().x + fullWidth + ImGui::GetStyle().ItemInnerSpacing.x, ImGui::GetCursorScreenPos().y + ImGui::GetStyle().FramePadding.y ),
        ImGui::GetColorU32( ImGuiCol_Text ),
        label
    );

    bool markEditedBecauseOfEnableSymmetry = false;

    ImVec2 baseCursorPos = ImGui::GetCursorPos();

    // The symmetry toggle.
    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + fullWidth - toggleSymButtonWidth );
    if ( ( buttonEx )( fmt::format( "{}###toggleSymmetry:{}", isAsym ? "\xC2\xB1"/*U+00B1 PLUS-MINUS SIGN*/ : "+/-", label ).c_str(), ImVec2( toggleSymButtonWidth, ImGui::GetFrameHeight() ), { .customTexture = UI::getTexture( UI::TextureType::GradientBtnGray ).get() } ) )
    {
        isAsym = !isAsym;

        StateStorage::writeBool( getStateKeyIsAsym(), isAsym );

        // If enabling symmetry, make the values equal.
        if ( !isAsym )
        {
            if ( !( plus == -minus ) ) // Not using `!=` to hopefully support NaNs a bit better, but this wasn't tested.
            {
                minus = -plus;
                markEditedBecauseOfEnableSymmetry = true; // Can't mark now, because we don't know which ID the `EndGroup()` will generate yet.
            }
        }
    }

    setTooltipIfHovered( isAsym ? "Make symmetric" : "Make asymmetric", scale() );

    // Restore cursor position.
    ImGui::SameLine(); // Just in case?
    ImGui::SetCursorPos( baseCursorPos );

    float widthWithoutButton = std::round( fullWidth - toggleSymButtonWidth - ImGui::GetStyle().ItemInnerSpacing.x );

    bool ret = markEditedBecauseOfEnableSymmetry;

    if ( isAsym )
    {
        // Two separate widths to avoid rounding errors. Should be almost equal.
        float halfWidth1 = std::round( ( widthWithoutButton - ImGui::GetStyle().ItemInnerSpacing.x ) / 2 );
        float halfWidth2 = std::round( widthWithoutButton - ImGui::GetStyle().ItemInnerSpacing.x - halfWidth1 );

        { // The group with two inputs.
            ImGui::BeginGroup();
            MR_FINALLY{ ImGui::EndGroup(); };

            // Adjust the unit params for plus.
            unitToStringParams.plusSign = true;

            { // The plus half.
                ImGui::PushItemWidth( halfWidth1 ); // This instead of `SetNextItemWidth()` just in case, in case the lambda uses it multiple times somehow.
                MR_FINALLY{ ImGui::PopItemWidth(); };
                if ( func( getPlusName().c_str(), plus, unitToStringParams ) )
                    ret = true;
            }

            ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );

            // Adjust the unit params for minus.
            unitToStringParams.zeroMode = ZeroMode::alwaysNegative;

            { // The minus half.
                ImGui::PushItemWidth( halfWidth2 ); // This instead of `SetNextItemWidth()` just in case, in case the lambda uses it multiple times somehow.
                MR_FINALLY{ ImGui::PopItemWidth(); };
                if ( func( getMinusName().c_str(), minus, std::move( unitToStringParams ) ) ) // Move the params on the last usage, in case the lambda decides to take them by value for some reason.
                    ret = true;
            }
        }
    }
    else
    {
        ImGui::PushItemWidth( widthWithoutButton ); // This instead of `SetNextItemWidth()` just in case, in case the lambda uses it multiple times somehow.
        MR_FINALLY{ ImGui::PopItemWidth(); };

        // Adjust the unit params.

        // Prepend plus-minus to the decoration format string.
        std::string decorationFormatStringStorage;
        if ( hasFormatPlaceholders( unitToStringParams.decorationFormatString ) )
        {
            decorationFormatStringStorage = "\xC2\xB1"; // U+00B1 PLUS-MINUS SIGN
            decorationFormatStringStorage += unitToStringParams.decorationFormatString;
            unitToStringParams.decorationFormatString = decorationFormatStringStorage; // This is a `string_view`! D:
        }

        // This name can't match the plus/minus names, because we check if those are active above.
        if ( func( fmt::format( "###sym:{}", label ).c_str(), plus, std::move( unitToStringParams ) ) ) // Move the params on the last usage, in case the lambda decides to take them by value for some reason.
        {
            minus = -plus;
            ret = true;
        }
    }

    if ( markEditedBecauseOfEnableSymmetry )
        detail::markItemEdited( ImGui::GetItemID() );

    return ret;
}

template <UnitEnum E, detail::Scalar T, detail::ValidDragSpeedForTargetType<T> SpeedType, typename F>
bool dragPlusMinus( const char* label, T& plus, T& minus, SpeedType speed, T plusMin, T plusMax, UnitToStringParams<E> unitParams, ImGuiSliderFlags flags, F&& wrapFunc )
{
    return plusMinusGeneric(
        label, plus, minus, std::move( unitParams ),
        [&]( const char* subLabel, T& value, const UnitToStringParams<E>& subUnitParams ) -> bool
        {
            return wrapFunc(
                subLabel,
                value,
                [&]() -> bool
                {
                    bool isPlus = &value == &plus;
                    return (drag)( subLabel, value, speed, isPlus ? plusMin : -plusMax, isPlus ? plusMax : -plusMin, subUnitParams, flags );
                }
            );
        }
    );
}

template <UnitEnum E, detail::Scalar T, typename F>
bool inputPlusMinus( const char* label, T& plus, T& minus, T plusMin, T plusMax, UnitToStringParams<E> unitParams, ImGuiSliderFlags flags, F&& wrapFunc )
{
    return plusMinusGeneric(
        label, plus, minus, std::move( unitParams ),
        [&]( const char* subLabel, T& value, const UnitToStringParams<E>& subUnitParams ) -> bool
        {
            return wrapFunc(
                subLabel,
                value,
                [&]() -> bool
                {
                    bool isPlus = &value == &plus;
                    return (input)( subLabel, value, isPlus ? plusMin : -plusMax, isPlus ? plusMax : -plusMin, subUnitParams, flags );
                }
            );
        }
    );
}

}
