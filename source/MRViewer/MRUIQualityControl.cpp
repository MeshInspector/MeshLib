#include "MRUIQualityControl.h"
#include "MRViewer/MRUIStyle.h"
#include "MRPch/MRSpdlog.h"

namespace MR::QualityControl
{

bool inputTolerance( const char* label, ObjectComparableWithReference& object, std::size_t index )
{
    auto toleranceOpt = object.getComparisonTolerence( index );

    ObjectComparableWithReference::ComparisonTolerance tolerance;
    if ( toleranceOpt )
        tolerance = *toleranceOpt;

    UnitToStringParams<LengthUnit> unitParams;
    if ( !toleranceOpt )
        unitParams.decorationFormatString = "\xE2\x80\x94"; // U+2014 EM DASH

    bool ret = false;

    if ( object.comparisonToleranceIsAlwaysOnlyPositive( index ) )
    {
        if ( UI::input<LengthUnit>( fmt::format( "###{}", label ).c_str(), tolerance.positive, 0.f, FLT_MAX, unitParams, UI::defaultSliderFlags ) )
        {
            assert( tolerance.negative == 0 );
            tolerance.negative = 0;

            object.setComparisonTolerance( index, tolerance );
            ret = true;
        }

        ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );

        // Draw the label separately to be able to color it separately.
        ImGui::TextUnformatted( label );
    }
    else
    {
        if ( UI::inputPlusMinus<LengthUnit>( label, tolerance.positive, tolerance.negative, 0.f, FLT_MAX, unitParams, UI::defaultSliderFlags,
            [&]( const char* subLabel, float& value, auto&& f )
            {
                (void)value;

                // Gray out this box if the input is nullopt. We also check that the current sub-input isn't focused; if it's focused, we always use the normal color.
                ImGui::PushStyleColor( ImGuiCol_Text, ImGui::GetStyleColorVec4( toleranceOpt || UI::isItemActive( subLabel ) ? ImGuiCol_Text : ImGuiCol_TextDisabled ) );
                MR_FINALLY{ ImGui::PopStyleColor(); };
                return f();
            }
        ) )
        {
            object.setComparisonTolerance( index, tolerance );
            ret = true;
        }
    }

    return ret;
}

}
