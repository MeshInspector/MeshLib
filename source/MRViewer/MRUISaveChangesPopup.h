#pragma once

#include <string>
#include <functional>

#include "exports.h"

namespace MR
{

namespace UI
{
// draw dilalog for save scene
// name - the signature of the dialog box
// label - the text description for which the action will be selected
// customFunction - the action when exiting the dialog box, if you confirm this action
// need to call ImGui::OpenPopup(name)
MRVIEWER_API void saveChangesPopup( float scaling, const std::string& name, const std::string& label, const std::function<void()>& customFunction );
}

}
