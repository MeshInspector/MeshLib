#pragma once

#include <string>
#include <functional>

#include "exports.h"

namespace MR
{

namespace UI
{

struct SettingsSaveChangesPopup
{
    // menu scaling
    float scaling = 1.0f;
    // the text for the button, if you need to exit the application
    std::string out = "Sign out";
    // text for button if need save scene
    std::string dontSave = "Don't Save";
};
// Shows ImGui popup that suggests user to save changes,
// user need to call ImGui::OpenPopup( str_id ) to open this popup.
// It has 3 options: save, don't save, cancel
//     str_id - ImGui string id for the popup window
//     header - header that is used in dialog
//     onOk - if not empty this function is called on "save" and "not save" options (if succeed)
MRVIEWER_API void saveChangesPopup( const char* str_id, const char* header, std::function<void()> onOk = {}, const SettingsSaveChangesPopup& settings = {} );
}

}
