#pragma once

#include <array>

namespace MR
{

enum class MouseButton
{
    Left = 0, // GLFW_MOUSE_BUTTON_1
    Right = 1, // GLFW_MOUSE_BUTTON_2
    Middle = 2, // GLFW_MOUSE_BUTTON_3
    Count,
    NoButton = Count
};

// Standard mouse functions for camera control
enum class MouseMode
{
    None,
    Rotation,       // Rotate camera around selected point
    Translation,    // Translate camera preserving its direction
    Roll,           // Rotate camera around axis orthogonal to screen
    Count
};

inline std::string getMouseModeString( MouseMode mode )
{
    constexpr std::array<const char*, size_t( MouseMode::Count )> names =
    {
        "None",
        "Rotation",
        "Translation",
        "Roll"
    };
    return names[int( mode )];
}

} //namespace MR
