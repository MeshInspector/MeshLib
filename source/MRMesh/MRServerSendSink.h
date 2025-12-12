#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

/// This class is provided as spdlog::base_sink to collect log messages
/// into WebLogger for later sending to remote logging service
class ServerSendSink : public spdlog::sinks::base_sink<spdlog::details::null_mutex>
{
private:
    virtual void sink_it_( const spdlog::details::log_msg& msg ) override;
    virtual void flush_() override { }
};

MRMESH_API void addServerSendSink();

} // namespace MR
