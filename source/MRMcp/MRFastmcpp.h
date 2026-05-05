#pragma once

// Centralised fastmcpp wrapper. Both MRMcp.cpp and MRMCPGateway.cpp must pull
// fastmcpp in before any standard library header (fastmcpp's macro tricks rely
// on that ordering). Keep `#undef _t` next to the includes: fastmcpp uses `_t`
// as a template parameter and our translation macro shadows it.

#undef _t

#if defined( __GNUC__ )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined( _MSC_VER )
#pragma warning( push )
#pragma warning( disable: 4100 ) // unreferenced formal parameter
#pragma warning( disable: 4355 ) // 'this': used in base member initializer list
#endif

#include <fastmcpp.hpp>
#include <fastmcpp/proxy.hpp>
#include <fastmcpp/server/sse_server.hpp>
#include <fastmcpp/server/stdio_server.hpp>
#include <fastmcpp/client/transports.hpp>
#include <fastmcpp/mcp/handler.hpp>

#if defined( __GNUC__ )
#pragma GCC diagnostic pop
#elif defined( _MSC_VER )
#pragma warning( pop )
#endif
