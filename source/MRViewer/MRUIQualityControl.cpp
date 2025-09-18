#include "MRUIQualityControl.h"

#include "MRViewer/MRUIStyle.h"

namespace MR::QualityControl
{

bool inputTolerance( const char* label, std::optional<ObjectComparableWithReference::ComparisonTolerance>& toleranceOpt )
{
    ObjectComparableWithReference::ComparisonTolerance tolerance;
    if ( toleranceOpt )
        tolerance = *toleranceOpt;

    UnitToStringParams<LengthUnit> unitParams;
    if ( !toleranceOpt )
        unitParams.decorationFormatString = "\xE2\x80\x94"; // U+2014 EM DASH

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
        toleranceOpt = tolerance;
        return true;
    }

    return false;
}

}
