#include "MRImGuiMultiViewport.h"

namespace ImGuiMV
{


void SetNextWindowPosMainViewport( const ImVec2& pos, ImGuiCond cond /*= 0*/, const ImVec2& pivot /*= ImVec2( 0, 0 )*/ )
{
    ImGui::SetNextWindowViewport( ImGui::GetMainViewport()->ID );
    ImVec2 winPos = ImGui::GetMainViewport()->Pos;
    ImGui::SetNextWindowPos( winPos + pos, cond, pivot );
}

ImVec2 GetLocalMousePos()
{
    return ImGui::GetMousePos() - ImGui::GetMainViewport()->Pos;
}

ImVec2 GetMainViewportShift()
{
    return ImGui::GetMainViewport()->Pos;
}

MR::Contour2f Screen2WindowPoints( const MR::Contour2f& points )
{
    const ImVec2 shiftMV = GetMainViewportShift();
    if ( shiftMV == ImVec2( 0, 0 ) )
        return points;

    MR::Contour2f windowPoints( points.size() );
    for ( int i = 0; i < points.size(); ++i )
        windowPoints[i] = points[i] - shiftMV;
    return windowPoints;
}

}
