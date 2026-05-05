#pragma once

#include "MRExpected.h"

namespace MR
{

/// in Debug configuration: simply executes given task, returns no error;
/// in Release configuration: executes given task inside C++ try-catch block, returns exception's info as error string;
///                           on Windows it also catches structured exceptions (SEH)
MRMESH_API Expected<void> protectedRun( const std::function<void ()>& task );

/// returns a function, which executes protectedRun( unprotectedFunc ), logging a possible exception but always returning
/// For example, executing in Release configuration the following line does not terminate the program:
///    std::thread( protectedFunc( []() { throw std::runtime_error( "Ops" ); } ) ).join();
MRMESH_API std::function<void ()> protectedFunc( std::function<void ()> unprotectedFunc );

} //namespace MR
