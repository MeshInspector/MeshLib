#pragma once
#include "imgui.h"
#include "exports.h"

namespace MR
{

namespace UI
{

/// init internal parameters
MRVIEWER_API void init();



/// draw gradient button, which can be disabled (active = false)
MRVIEWER_API bool button( const char* label, bool active, const Vector2f& size = Vector2f( 0, 0 ) );
/// draw gradient button
/// returns true if button is clicked in this frame, or key is pressed (optional)
MRVIEWER_API bool button( const char* label, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None );
/// draw gradient button with the ordinary button size
/// returns true if button is clicked in this frame, or key is pressed (optional)
inline bool buttonCommonSize( const char* label, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None )
{
    return button( label, { size.x, size.y == 0.0f ? ImGui::GetFrameHeight() : size.y }, key );
}


/// draw gradient checkbox
MRVIEWER_API bool checkbox( const char* label, bool* value );
/// draw gradient checkbox with mixed state
MRVIEWER_API bool checkboxMixed( const char* label, bool* value, bool mixed );
/// draw gradient checkbox
template<typename Getter, typename Setter>
static bool checkbox( const char* label, Getter get, Setter set )
{
    bool value = get();
    bool ret = checkbox( label, &value );
    set( value );
    return ret;
}


} // namespace UI

}
