#include "MRImGuiMultiViewport.h"

namespace MR::ImGuiMV
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

ImVec2 Screen2WindowSpaceImVec2( const ImVec2& point )
{
    return point - GetMainViewportShift();
}

Contour2f Screen2WindowSpaceContour2f( const Contour2f& points )
{
    const ImVec2 shiftMV = GetMainViewportShift();
    if ( shiftMV == ImVec2( 0, 0 ) )
        return points;

    Contour2f windowPoints( points.size() );
    for ( int i = 0; i < points.size(); ++i )
        windowPoints[i] = points[i] - shiftMV;
    return windowPoints;
}

ImVec2 Window2ScreenSpaceImVec2( const ImVec2& point )
{
    return point + GetMainViewportShift();
}

}
