#pragma once
#include "exports.h"
#include "imgui.h"
#include <string>
#include <optional>

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
/// draw gradient checkbox
/// if valid is false checkbox is disabled
MRVIEWER_API bool checkboxValid( const char* label, bool* value, bool valid );
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
MRVIEWER_API bool colorEdit4( const char* label, Vector4f& color, ImGuiColorEditFlags flags = ImGuiColorEditFlags_None );

/// draw combo box
MRVIEWER_API bool combo( const char* label, int* v, const std::vector<std::string>& options,
    bool showPreview = true, const std::vector<std::string>& tooltips = {}, const std::string& defaultText = "Not selected" );



/// draw input text box with text aligned by center
MRVIEWER_API bool inputTextCentered( const char* label, std::string& str, float width = 0.0f, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL );

/// draw read-only text box with text aligned by center
MRVIEWER_API void inputTextCenteredReadOnly( const char* label, const std::string& str, float width = 0.0f, const std::optional<ImVec4>& textColor = {} );

/// similar to ImGui::Text but use current text color with alpha channel = 0.5
MRVIEWER_API void transparentText( const char* fmt, ... );
/// similar to ImGui::TextWrapped but use current text color with alpha channel = 0.5
MRVIEWER_API void transparentTextWrapped( const char* fmt, ... );

/// draw tooltip only if current item is hovered
MRVIEWER_API void setTooltipIfHovered( const std::string& text, float scaling );

/// add text with separator line 
/// if issueCount is greater than zero, this number will be displayed in red color after the text. 
/// If it equals zero - in green color
/// Otherwise it will not be displayed
MRVIEWER_API void separator( float scaling, const std::string& text = "", int issueCount = -1 );

/// draws progress bar
/// note that even while scaling is given by argument size should still respect it
/// size: x <= 0 - take all available width
///       y <= 0 - frame height
MRVIEWER_API void progressBar( float scaling, float fraction, const Vector2f& size = Vector2f( -1, 0 ) );

} // namespace UI

}
