#include "MRRenderClickableRect.h"

#include "MRViewer/MRImGuiVectorOperators.h"

namespace MR
{

void BasicClickableRectUiRenderTask::earlyBackwardPass( const BackwardPassParams& backParams )
{
    if ( bool( backParams.consumedInteractions & InteractionMask::mouseHover ) )
        return;

    bool nextIsHovered = false;
    isActive = false;

    // If the rect has nonzero area...
    if ( ImGuiMath::CompareAll( clickableCornerA ) < clickableCornerB )
    {
        // React to hover and possibly click.
        if ( ImGuiMath::CompareAll( ImGui::GetMousePos() ) >= clickableCornerA && ImGuiMath::CompareAll( ImGui::GetMousePos() ) < clickableCornerB )
        {
            if ( backParams.tryConsumeMouseHover() && enabled ) // In this order!
            {
                nextIsHovered = true;

                // Note, we check `isHovered` from the previous frame.
                // We do this to prevent bugs caused by clicking the rect on the same frame as it appears.

                if ( isHovered && ImGui::IsMouseDown( ImGuiMouseButton_Left ) )
                    isActive = true;

                if ( isHovered && ImGui::IsMouseClicked( ImGuiMouseButton_Left ) )
                    onClick();
            }
        }
    }

    isHovered = nextIsHovered;
}

}
