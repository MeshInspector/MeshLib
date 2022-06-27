#pragma once
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
