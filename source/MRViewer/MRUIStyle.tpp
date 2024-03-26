#pragma once

// A template implementation file for `MRUIStyle.h`. Include that file.

#include "MRUIStyle.h" // To help intellisense.

namespace MR::UI
{

namespace detail
{
    // The `ImGuiDataType_??` constant for the given type.
    template <Scalar T>
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

    template <UnitEnum E, VectorOrScalar T, typename F>
    [[nodiscard]] bool unitWidget( const char* label, T& v, UnitToStringParams<E>& unitParams, F&& func )
    {
        const std::optional<E> originalSourceUnit = unitParams.sourceUnit;

        // We need to jump through some hoops to handle integers.
        // If the input is an integer and if no conversion is needed, we draw the widget directly for that integer.
        // If the input is an integer and we do need a conversion, we draw a floating-point slider, then round back.

        const bool mustConvertUnits =
            unitParams.sourceUnit &&
            *unitParams.sourceUnit != unitParams.targetUnit &&
            getUnitInfo( *unitParams.sourceUnit ).conversionFactor != getUnitInfo( unitParams.targetUnit ).conversionFactor;

        constexpr bool targetIsIntegral = std::is_integral_v<typename VectorTraits<T>::BaseType>;

        auto drawWidget = [&]<VectorOrScalar U>( U& value ) -> bool
        {
            bool ret = false;
            if constexpr ( VectorTraits<T>::size == 1 )
            {
                ret = std::forward<F>( func )( label, value, 0 );
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

                        if constexpr ( !std::is_integral_v<typename VectorTraits<U>::BaseType> )
                        {
                            // Convert back to source units.
                            if ( mustConvertUnits )
                                VectorTraits<T>::getElem( i, value ) = convertUnits( unitParams.targetUnit, *originalSourceUnit, VectorTraits<T>::getElem( i, value ) );

                            if constexpr ( targetIsIntegral )
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
            valueCopy = convertUnits( unitParams.sourceUnit.value_or( unitParams.targetUnit ), unitParams.targetUnit, v );
            unitParams.sourceUnit.reset(); // To prevent the user from performing unnecessary conversions.
            return drawWidget( valueCopy );
        }

        return drawWidget( v );
    }

    template <UnitEnum E, VectorOrScalar T>
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

    template <UnitEnum E, VectorOrScalar T>
    [[nodiscard]] T getDefaultStep( bool fast )
    {
        if constexpr ( std::is_integral_v<T> )
            return fast ? 100 : 1;
        else
            return 0;
    }

    template <UnitEnum E, VectorOrScalar T>
    [[nodiscard]] std::string getDragRangeTooltip( T min, T max, const UnitToStringParams<E>& unitParams )
    {
        bool haveMin = min > std::numeric_limits<T>::lowest();
        bool haveMax = max < std::numeric_limits<T>::max();

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
}

template <UnitEnum E, detail::VectorOrScalar T, detail::ValidBoundForTargetType<T> U>
bool slider( const char* label, T& v, const U& vMin, const U& vMax, UnitToStringParams<E> unitParams, ImGuiSliderFlags flags )
{
    // Adjust the parameters:
    // Don't strip trailing zeroes, otherwise the numbers jump too much.
    unitParams.stripTrailingZeroes = false;

    auto fixedMin = convertUnits( unitParams.sourceUnit.value_or( unitParams.targetUnit ), unitParams.targetUnit, vMin );
    auto fixedMax = convertUnits( unitParams.sourceUnit.value_or( unitParams.targetUnit ), unitParams.targetUnit, vMax );

    if ( !unitsAreEquivalent( unitParams.sourceUnit.value_or( unitParams.targetUnit ), unitParams.targetUnit ) )
    {
        // Without this flag the value changes slightly when you release the mouse.
        flags |= ImGuiSliderFlags_NoRoundToFormat;
    }

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
                elemMin = &VectorTraits<U>::getElem( i, fixedMin );
                elemMax = &VectorTraits<U>::getElem( i, fixedMax );
            }

            return detail::genericSlider(
                elemLabel, detail::imGuiTypeEnum<ElemType>(), &elemVal, elemMin, elemMax, valueToString<E>( elemVal, unitParams ).c_str(), flags
            );
        } );
}

template <UnitEnum E, detail::VectorOrScalar T, detail::ValidDragSpeedForTargetType<T> SpeedType, detail::ValidBoundForTargetType<T> U>
bool drag( const char* label, T& v, SpeedType vSpeed, const U& vMin, const U& vMax, UnitToStringParams<E> unitParams, ImGuiSliderFlags flags )
{
    // Adjust the parameters:
    // Don't strip trailing zeroes, otherwise the numbers jump too much.
    unitParams.stripTrailingZeroes = false;

    auto fixedSpeed = convertUnits( unitParams.sourceUnit.value_or( unitParams.targetUnit ), unitParams.targetUnit, vSpeed );
    auto fixedMin = convertUnits( unitParams.sourceUnit.value_or( unitParams.targetUnit ), unitParams.targetUnit, vMin );
    auto fixedMax = convertUnits( unitParams.sourceUnit.value_or( unitParams.targetUnit ), unitParams.targetUnit, vMax );

    if ( !unitsAreEquivalent( unitParams.sourceUnit.value_or( unitParams.targetUnit ), unitParams.targetUnit ) )
    {
        // Without this flag the value changes slightly when you release the mouse.
        flags |= ImGuiSliderFlags_NoRoundToFormat;
    }

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
                elemMin = &VectorTraits<U>::getElem( i, fixedMin );
                elemMax = &VectorTraits<U>::getElem( i, fixedMax );
            }

            bool ret = ImGui::DragScalar(
                elemLabel, detail::imGuiTypeEnum<ElemType>(), &elemVal,
                float( VectorTraits<SpeedType>::getElem( i, fixedSpeed ) ), elemMin, elemMax, valueToString<E>( elemVal, unitParams ).c_str(), flags
            );
            detail::drawDragTooltip( detail::getDragRangeTooltip( VectorTraits<U>::getElem( i, fixedMin ), VectorTraits<U>::getElem( i, fixedMax ), unitParams ) );
            return ret;
        } );
}

template <UnitEnum E, detail::VectorOrScalar T, detail::ValidBoundForTargetType<T> U>
bool input( const char* label, T& v, const U& vStep, const U& vStepFast, UnitToStringParams<E> unitParams, ImGuiInputTextFlags flags )
{
    auto fixedStep = convertUnits( unitParams.sourceUnit.value_or( unitParams.targetUnit ), unitParams.targetUnit, vStep );
    auto fixedStepFast = convertUnits( unitParams.sourceUnit.value_or( unitParams.targetUnit ), unitParams.targetUnit, vStepFast );

    return detail::unitWidget( label, v, unitParams,
        [&]<typename ElemType>( const char* elemLabel, ElemType& elemVal, int i )
        {
            const ElemType *elemStep = nullptr;
            const ElemType *elemStepFast = nullptr;

            if constexpr ( std::is_integral_v<ElemType> )
            {
                elemStep = &VectorTraits<U>::getElem( i, vStep );
                elemStepFast = &VectorTraits<U>::getElem( i, vStepFast );
            }
            else
            {
                elemStep = &VectorTraits<U>::getElem( i, fixedStep );
                elemStepFast = &VectorTraits<U>::getElem( i, fixedStepFast );
            }

            return ImGui::InputScalar(
                elemLabel, detail::imGuiTypeEnum<ElemType>(), &elemVal,
                // When the step is zero, pass nullptr to disable the buttons.
                *elemStep ? elemStep : nullptr,  *elemStepFast ? elemStepFast : nullptr,
                valueToString<E>( elemVal, unitParams ).c_str(), flags
            );
        } );
}

template <UnitEnum E, detail::VectorOrScalar T>
void readOnlyValue( const char* label, const T& v, std::optional<ImVec4> textColor, UnitToStringParams<E> unitParams )
{
    (void)detail::unitWidget( label, const_cast<T&>( v ), unitParams,
        [&]( const char* elemLabel, auto& elemVal, int i )
        {
            (void)i;
            inputTextCenteredReadOnly( elemLabel, valueToString<E>( elemVal, unitParams ), ImGui::CalcItemWidth(), textColor );
            return false;
        } );
}


}
