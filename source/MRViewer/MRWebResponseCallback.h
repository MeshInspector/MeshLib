#pragma once

#include "MRViewerFwd.h"
#include <json/forwards.h>

namespace MR
{

using WebResponseCallback = std::function<void( const Json::Value& response )>;

} //namespace MR
