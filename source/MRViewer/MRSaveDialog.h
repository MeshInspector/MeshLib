#pragma once

#include <string>
#include <functional>

#include "exports.h"

namespace MR
{
// draw dilalog for save scene
MRVIEWER_API void saveSceneDialog( float scaling, const std::string& name, const std::string& label, const std::function<void()>& customFunction );

}
