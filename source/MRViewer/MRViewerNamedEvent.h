#pragma once

#include "MRViewerFwd.h"
#include <functional>
#include <string>

namespace MR
{

using EventCallback = std::function<void()>;
struct ViewerNamedEvent
{
    std::string name;
    EventCallback cb;
};

} //namespace MR
