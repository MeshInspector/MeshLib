#include "MRRenderNameObject.h"
#include "MRColorTheme.h"
#include "MRImGuiVectorOperators.h"
#include "MRRibbonMenu.h"
#include "MRViewport.h"

#include "MRMesh/MRFinally.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRVisualObject.h"

#include <imgui.h>

namespace MR
{

void RenderNameObject::Task::earlyBackwardPass( const BackwardPassParams& backParams )
{
    if ( bool( backParams.consumedInteractions & InteractionMask::mouseHover ) )
        return;

    // If it wasn't clipped to nothing...
    if ( ImGuiMath::CompareAll( windowCornerA ) < windowCornerB )
    {
        // React to hover and possibly click.
        if ( ImGuiMath::CompareAll( ImGui::GetMousePos() ) >= windowCornerA && ImGuiMath::CompareAll( ImGui::GetMousePos() ) < windowCornerB )
        {
            if ( backParams.tryConsumeMouseHover() )
            {
                isHovered = true;

                if ( ImGui::IsMouseDown( ImGuiMouseButton_Left ) )
                    isActive = true;

                if ( ImGui::IsMouseClicked( ImGuiMouseButton_Left ) )
                {
                    RibbonMenu::instance()->simulateNameTagClick(
                        // Yes, a dumb cast. We could find the same object in the scene, but it's a waste of time.
                        // Changing the `RenderObject` constructor parameter to accept a non-const reference requires changing a lot of stuff.
                        *const_cast<VisualObject*>( object ),
                        ImGui::GetIO().KeyCtrl ? ImGuiMenu::NameTagSelectionMode::toggle : ImGuiMenu::NameTagSelectionMode::selectOne
                    );
                }
            }
        }
    }
}

void RenderNameObject::Task::renderPass()
{
    const float
        // Button rounding.
        rounding = 4 * params->scale,
        lineWidth = 2 * params->scale,
        lineOutlineWidth = 1 * params->scale,
        buttonOutlineWidth = 1 * params->scale;

    const ImU32 colorOutline = ImGui::ColorConvertFloat4ToU32( ImVec4( 0, 0, 0, 0.5f ) );

    ImVec4 colorMainFloat = ImVec4( object->getFrontColor( object->isSelected() ) );
    colorMainFloat = ImGuiMath::mix( 0.1f, colorMainFloat, ImVec4( 0, 0, 0, 1 ) );
    colorMainFloat.w = 1;
    ImU32 colorMain = ImGui::ColorConvertFloat4ToU32( colorMainFloat );

    ImVec4 colorHoveredFloat = ImVec4( object->getFrontColor( object->isSelected() ) );
    colorHoveredFloat = ImGuiMath::mix( 0.2f, colorHoveredFloat, ImVec4( 0, 0, 0, 1 ) );
    colorHoveredFloat.w = 1;
    ImU32 colorHovered = ImGui::ColorConvertFloat4ToU32( colorHoveredFloat );

    // A crude conversion to grayscale. Good enough for our purposes.
    float colorMainLuminosity = 0.2126f * colorMainFloat.x + 0.7152f * colorMainFloat.y + 0.0722f * colorMainFloat.z;

    ImU32 colorText = ImGui::ColorConvertFloat4ToU32( colorMainLuminosity > 0.5f ? ImVec4( 0, 0, 0, 1 ) : ImVec4( 1, 1, 1, 1 ) );

    ImDrawList& drawList = *ImGui::GetBackgroundDrawList();

    auto makeLinePath = [&]( bool isOutline )
    {
        if ( point != point2 || point2 != textCenter )
        {
            ImVec2 firstPointOffset;
            if ( isOutline )
            {
                if ( point != point2 )
                    firstPointOffset = ImGuiMath::normalize( point - point2 );
                else
                    firstPointOffset = ImGuiMath::normalize( point2 - textCenter );

                firstPointOffset *= lineOutlineWidth * 0.5f; // An arbitrary factor.
            }

            drawList.PathLineTo( point + firstPointOffset );
            if ( point2 != point )
                drawList.PathLineTo( point2 );
            if ( textCenter != point2 )
                drawList.PathLineTo( textCenter );
        }
    };

    // Line outline.
    makeLinePath( true );
    drawList.PathStroke( colorOutline, 0, lineWidth + lineOutlineWidth * 2 );

    // Button outline.
    drawList.AddRectFilled( textPos - paddingA - buttonOutlineWidth, textPos + textSize + paddingB + buttonOutlineWidth, colorOutline, rounding + buttonOutlineWidth );

    // The line itself.
    makeLinePath( false );
    drawList.PathStroke( colorMain, 0, lineWidth );

    // The background rect.
    drawList.AddRectFilled( textPos - paddingA, textPos + textSize + paddingB, isHovered && !isActive ? colorHovered : colorMain, rounding );

    // The text.
    // Split to segments to center each line individually.
    float textY = textPos.y;
    split( text, "\n", [&]( std::string_view segment )
    {
        ImVec2 pos( std::round( textPos.x + ( textSize.x - ImGui::CalcTextSize( segment.data(), segment.data() + segment.size() ).x ) / 2 ), textY );
        drawList.AddText( pos, colorText, segment.data(), segment.data() + segment.size() );
        textY += ImGui::GetTextLineHeight();
        return false;
    } );

    // Reset the variables.
    isHovered = false;
    isActive = false;
}

void RenderNameObject::renderUi( const UiRenderParams& params )
{
    task_.params = &params;

    task_.isHovered = false;
    task_.isActive = false;

    if ( !task_.object->getVisualizeProperty( VisualizeMaskType::Name, params.viewportId ) )
        return; // The name is hidden in this viewport.

    const float
        // When offsetting the button relative to a point, this is the gap to the point (or rather to an imaginary line passing through the point,
        //   perpendicular to the offset direction).
        buttonSpacingToPoint = 30 * params.scale;


    task_.paddingA = ImGuiMath::round( ImVec2( 4, 2 ) * params.scale ),
    task_.paddingB = ImGuiMath::round( ImVec2( 4, 4 ) * params.scale );

    auto xf = task_.object->worldXf();

    Vector3f localPoint = nameUiPoint;

    Vector3f worldPoint = xf( localPoint );
    Vector3f worldPoint2 = xf( localPoint + nameUiLocalOffset );
    ImVec2 point3Offset = ImVec2( nameUiScreenOffset ) * params.scale;

    task_.text = getObjectNameString( *task_.object, params.viewportId );
    task_.textSize = ImGui::CalcTextSize( task_.text.c_str() );

    Viewport& viewportRef = Viewport::get( params.viewportId );

    ImVec2 viewportCornerA( float( params.viewport.x ), float( ImGui::GetIO().DisplaySize.y - params.viewport.y - params.viewport.w ) );
    ImVec2 viewportCornerB( float( params.viewport.x + params.viewport.z ), float( ImGui::GetIO().DisplaySize.y - params.viewport.y ) );

    auto toScreenCoords = [&]( Vector3f point, float* depthOutput = nullptr ) -> ImVec2
    {
        auto result = viewportRef.projectToViewportSpace( point );
        if ( depthOutput )
            *depthOutput = result.z;
        return ImVec2( result.x, result.y ) + viewportCornerA;
    };

    Vector3f fixedWorldPoint = worldPoint;
    Vector3f fixedWorldPoint2 = worldPoint2;

    if ( nameUiRotateToScreenPlaneAroundSphereCenter )
    {
        Vector3f center = xf( *nameUiRotateToScreenPlaneAroundSphereCenter );

        Vector3f delta = fixedWorldPoint - center;
        float deltaLen = delta.lengthSq();
        if ( deltaLen > 0 )
        {
            deltaLen = std::sqrt( deltaLen );

            Vector3f dirTowardsCamera = Vector3f( params.viewMatrix.z.x, params.viewMatrix.z.y, params.viewMatrix.z.z ).normalized();

            Vector3f fixedDelta;
            fixedDelta = delta - dot( delta, dirTowardsCamera ) * dirTowardsCamera;
            fixedDelta = fixedDelta.normalized() * deltaLen;
            fixedWorldPoint = center + fixedDelta;

            fixedWorldPoint2 = center +
                Matrix3f::rotation( delta, fixedDelta ) * ( fixedWorldPoint2 - center );
        }
    }

    task_.point = toScreenCoords( fixedWorldPoint );
    task_.point2 = toScreenCoords( fixedWorldPoint2, &task_.renderTaskDepth );

    if ( nameUiRotateLocalOffset90Degrees )
        task_.point2 = task_.point + ImGuiMath::rot90( task_.point2 - task_.point );

    ImVec2 point3 = task_.point2 + point3Offset;

    ImVec2 lastSegmentOffset;
    if ( point3Offset != ImVec2{} )
        lastSegmentOffset = point3Offset;
    else if ( ImVec2 d = point3 - task_.point; d != ImVec2{} )
        lastSegmentOffset = d;

    task_.textCenter = point3;
    task_.textPos = task_.textCenter - task_.textSize / 2;

    if ( lastSegmentOffset != ImVec2{} )
    {
        ImVec2 prevPoint = point3 - lastSegmentOffset;

        ImVec2 pointA = task_.textPos - task_.paddingA - buttonSpacingToPoint;
        ImVec2 pointB = task_.textPos + task_.textSize + task_.paddingB + buttonSpacingToPoint;

        if ( ImGuiMath::CompareAll( prevPoint ) >= pointA && ImGuiMath::CompareAll( prevPoint ) < pointB )
        {
            ImVec2 offsetDir = ImGuiMath::normalize( lastSegmentOffset );

            ImVec2 absOffset(
                std::abs( ( offsetDir.x < 0 ? pointB.x : offsetDir.x > 0 ? pointA.x : 0 ) - prevPoint.x ),
                std::abs( ( offsetDir.y < 0 ? pointB.y : offsetDir.y > 0 ? pointA.y : 0 ) - prevPoint.y )
            );
            if ( offsetDir.x == 0 || ( std::abs( absOffset.y / offsetDir.y ) < std::abs( absOffset.x / offsetDir.x ) ) )
                absOffset.x = std::abs( absOffset.y / offsetDir.y * offsetDir.x );
            else
                absOffset.y = std::abs( absOffset.x / offsetDir.x * offsetDir.y );

            ImVec2 delta;

            if ( offsetDir.x > 0 )
                delta.x = absOffset.x;
            else if ( offsetDir.x < 0 )
                delta.x = -absOffset.x;

            if ( offsetDir.y > 0 )
                delta.y = absOffset.y;
            else if ( offsetDir.y < 0 )
                delta.y = -absOffset.y;

            task_.textPos += delta;
            task_.textCenter += delta;
        }
    }

    task_.textPos = ImGuiMath::round( task_.textPos );

    task_.windowCornerA = ImGuiMath::clamp( task_.textPos - task_.paddingA, viewportCornerA, viewportCornerB );
    task_.windowCornerB = ImGuiMath::clamp( task_.textPos + task_.textSize + task_.paddingB, viewportCornerA, viewportCornerB );

    // A non-owning pointer to our task_.
    params.tasks->push_back( { std::shared_ptr<void>{}, &task_ } );
}

std::string RenderNameObject::getObjectNameString( const VisualObject& object, ViewportId viewportId ) const
{
    (void)viewportId;
    return object.name();
}

}
