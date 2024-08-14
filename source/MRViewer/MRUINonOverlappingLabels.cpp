#include "MRUINonOverlappingLabels.h"

#include "MRViewer/MRUIRectAllocator.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"

namespace MR::UI
{

NonOverlappingLabelTask::NonOverlappingLabelTask()
{}

NonOverlappingLabelTask::NonOverlappingLabelTask( Params params )
    : params_( std::move( params ) )
{}

void NonOverlappingLabelTask::earlyBackwardPass( const BackwardPassParams& params )
{
    (void)params;

    std::optional<ImGuiMeasurementIndicators::Params> indParamsStorage;
    const ImGuiMeasurementIndicators::Params* indParams = params_.indicatorParams.get();
    if ( !indParams )
        indParams = &indParamsStorage.emplace();

    ImVec2 rectSize =
        ImGui::CalcTextSize( params_.text.data(), params_.text.data() + params_.text.size() ) +
        indParams->textOutlineWidth * 2 +
        indParams->textToLineSpacingA + indParams->textToLineSpacingB +
        indParams->textToLineSpacingRadius;
    Vector3f screenPos = getViewerInstance().viewportToScreen( getViewerInstance().viewport( params_.viewportId ).projectToViewportSpace( params_.worldPos ), params_.viewportId );
    renderTaskDepth = screenPos.z;
    // Since we save the center of the rect, any potential inaccuracies (though we should have none) in rect size calculation don't really matter.
    finalScreenPos_ = getDefaultLabelRectAllocator().createRect( params_.viewportId, params_.id, ImVec2( screenPos.x - rectSize.x / 2, screenPos.y - rectSize.y / 2 ), rectSize ) + rectSize / 2;
}

void NonOverlappingLabelTask::renderPass()
{
    std::optional<ImGuiMeasurementIndicators::Params> indParamsStorage;
    const ImGuiMeasurementIndicators::Params* indParams = params_.indicatorParams.get();
    if ( !indParams )
        indParams = &indParamsStorage.emplace();

    ImGuiMeasurementIndicators::text( ImGuiMeasurementIndicators::Element::both, params_.menuScaling, *indParams, finalScreenPos_, params_.text );
}

}
