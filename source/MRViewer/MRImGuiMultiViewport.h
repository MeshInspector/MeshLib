#pragma once
#include "exports.h"
#include "MRImGui.h"
#include "MRImGuiVectorOperators.h"

// namespace for easy access to functions related to ImGui MultiViewport
namespace ImGuiMV
{

// attaches the next window to the viewport of the main window and sets the position relative to it
MRVIEWER_API void SetNextWindowPosMainViewport( const ImVec2& pos, ImGuiCond cond = 0, const ImVec2& pivot = ImVec2( 0, 0 ) );

// returns the coordinates of the mouse in the main viewport space
MRVIEWER_API ImVec2 GetLocalMousePos();

// returns the shift of the main viewport relative to global coordinates
MRVIEWER_API ImVec2 GetMainViewportShift();

}
