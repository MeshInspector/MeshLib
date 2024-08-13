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
{
    if ( !params_.indicatorParams )
        params_.indicatorParams = std::make_shared<ImGuiMeasurementIndicators::Params>();
}

void NonOverlappingLabelTask::earlyBackwardPass( const BackwardPassParams& params )
{
    (void)params;
    ImVec2 rectSize =
        ImGui::CalcTextSize( params_.text.data(), params_.text.data() + params_.text.size() ) +
        params_.indicatorParams->textOutlineWidth * 2 +
        params_.indicatorParams->textToLineSpacingA + params_.indicatorParams->textToLineSpacingB +
        params_.indicatorParams->textToLineSpacingRadius;
    Vector3f screenPos = getViewerInstance().viewportToScreen( getViewerInstance().viewport( params_.viewportId ).projectToViewportSpace( params_.worldPos ), params_.viewportId );
    renderTaskDepth = screenPos.z;
    // Since we save the center of the rect, any potential inaccuracies (though we should have none) in rect size calculation don't really matter.
    finalScreenPos_ = getDefaultLabelRectAllocator().createRect( params_.viewportId, params_.id, ImVec2( screenPos.x - rectSize.x / 2, screenPos.y - rectSize.y / 2 ), rectSize ) + rectSize / 2;
}

void NonOverlappingLabelTask::renderPass()
{
    ImGuiMeasurementIndicators::text( ImGuiMeasurementIndicators::Element::both, params_.menuScaling, *params_.indicatorParams, finalScreenPos_, params_.text );
}

}
