#pragma once
#include "MRViewerFwd.h"
#include <json/forwards.h>

namespace MR
{
// Accumulate system information in Json value
MRVIEWER_API Json::Value GetSystemInfoJson();
}