#pragma once

#include "MRExpected.h"

namespace MR
{

/// in Debug configuration: simply executes given task, returns no error;
/// in Release configuration: executes given task inside C++ try-catch block, returns exception's info as error string;
MRMESH_API Expected<void> protectedRun( const std::function<void ()>& task );

/// on Windows in Release configuration: executes given task capturing first structured than C++ exceptions, returns exception's info as error string;
/// otherwise same as protectedRun
MRMESH_API Expected<void> protectedRunEx( const std::function<void ()>& task );

} //namespace MR
