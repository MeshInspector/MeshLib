#pragma once
#include "MRViewerFwd.h"
#include "MRPch/MRJson.h"

namespace MR
{
// Accumulate system information in Json value
MRVIEWER_API Json::Value GetSystemInfoJson();
}