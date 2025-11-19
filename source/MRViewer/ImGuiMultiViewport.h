#pragma once
#include "exports.h"
#include "MRImGui.h"
#include "MRImGuiVectorOperators.h"

// namespace for easy access to functions related to ImGui MultiViewport
namespace ImGuiMV
{

MRVIEWER_API void SetNextWindowPosMainViewport( const ImVec2& pos, ImGuiCond cond = 0, const ImVec2& pivot = ImVec2( 0, 0 ) );

MRVIEWER_API ImVec2 GetLocalMousePos();

MRVIEWER_API ImVec2 GetMainViewportShift();

}
