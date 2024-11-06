#pragma once

#include <string>
#include <functional>

#include "exports.h"

namespace MR
{

namespace UI
{

struct SaveChangesPopupSettings
{
    // menu scaling
    float scaling = 1.0f;
    // text that is shown if we have nothing to save
    std::string shortCloseText = "Close";
    // text that is shown if we have changes but don't want to save them
    std::string dontSaveText = "Don't Save";

    std::string saveTooltip = "Save current scene";
    std::string dontSaveTooltip = "Donh't save current scene";
    std::string cancelTooltip = "Cansel";
    // header that is used in dialog
    std::string header;
    // if not empty this function is called on "save" and "not save" options( if succeed )
    std::function<void()> onOk = {};
};
// Shows ImGui popup that suggests user to save changes,
// user need to call ImGui::OpenPopup( str_id ) to open this popup.
// It has 3 options: save, don't save, cancel
//     str_id - ImGui string id for the popup window
//     settings - settings for dialog
MRVIEWER_API void saveChangesPopup( const char* str_id, const SaveChangesPopupSettings& settings = {} );
}

}
