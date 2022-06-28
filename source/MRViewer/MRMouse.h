#pragma once

#include <array>

namespace MR
{

enum class MouseButton
{
    Left = 0, // GLFW_MOUSE_BUTTON_1
    Right = 1, // GLFW_MOUSE_BUTTON_2
    Middle = 2, // GLFW_MOUSE_BUTTON_3
    Count
};

enum class MouseMode
{
    None, Rotation, Translation, Count
};

inline std::string getMouseModeString( MouseMode mode )
{
    constexpr std::array<const char*, size_t( MouseMode::Count )> names =
    {
        "None",
        "Rotation",
        "Translation"
    };
    return names[int( mode )];
}

} //namespace MR
