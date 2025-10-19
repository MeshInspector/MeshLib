#pragma once

#include "MRMeshFwd.h"
#include "MRSignal.h"
#include <string>

namespace MR
{

/// activate this signal if you want to add some string in telemetry
extern MRMESH_API Signal<void( const std::string& )> TelemetrySignal;

} //namespace MR
