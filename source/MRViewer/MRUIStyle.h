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
MRVIEWER_API bool buttonEx( const char* label, bool active, const Vector2f& size = Vector2f( 0, 0 ), ImGuiButtonFlags flags = ImGuiButtonFlags_None );
/// draw gradient button, which can be disabled (active = false)
/// returns true if button is clicked in this frame, or key is pressed (optional)
MRVIEWER_API bool button( const char* label, bool active, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None );
/// draw gradient button
/// returns true if button is clicked in this frame, or key is pressed (optional)
inline bool button( const char* label, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None )
{
    return button( label, true, size, key );
}
/// draw gradient button with the ordinary button size
/// returns true if button is clicked in this frame, or key is pressed (optional)
MRVIEWER_API bool buttonCommonSize( const char* label, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None );



/// draw gradient checkbox
MRVIEWER_API bool checkbox( const char* label, bool* value );
/// draw gradient checkbox with mixed state
MRVIEWER_API bool checkboxMixed( const char* label, bool* value, bool mixed );
/// draw gradient checkbox
template<typename Getter, typename Setter>
inline bool checkbox( const char* label, Getter get, Setter set )
{
    bool value = get();
    bool ret = checkbox( label, &value );
    set( value );
    return ret;
}


/// draw gradient radio button
MRVIEWER_API bool radioButton( const char* label, int* value, int valButton );


/// draw gradient color edit 4
MRVIEWER_API bool colorEdit4( const char* label, Vector4f& color, ImGuiColorEditFlags flags /*= ImGuiColorEditFlags_None*/ );

/// draw combo box
MRVIEWER_API bool combo( const char* label, int* v, const std::vector<std::string>& options,
    bool showPreview = true, const std::vector<std::string>& tooltips = {}, const std::string& defaultText = "Not selected" );

} // namespace UI

}
