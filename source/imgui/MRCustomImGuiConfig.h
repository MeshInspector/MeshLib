#pragma once

#include <MRMesh/MRVector2.h>

#ifdef _WIN32
#   ifdef MRIMGUI_EXPORT
#       define IMGUI_API __declspec(dllexport)
#   else
#       define IMGUI_API __declspec(dllimport)
#   endif
#else
#       define IMGUI_API __attribute__((visibility("default")))
#endif

struct ImGuiContext;

IMGUI_API ImGuiContext*& MyImGuiTLS();
#define GImGui MyImGuiTLS()

#define IM_VEC2_CLASS_EXTRA \
    constexpr ImVec2( const MR::Vector2f & v ) noexcept : x( v.x ), y( v.y ) {} \
    constexpr operator MR::Vector2f() const noexcept { return { x, y }; }
