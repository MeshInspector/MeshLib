#include "MRServerSendSink.h"
#include "MRLog.h"
#include "MRWebLogger.h"

namespace MR
{

void ServerSendSink::sink_it_( const spdlog::details::log_msg& msg )
{
    std::string message = std::string( msg.payload.data(), msg.payload.size() );
    WebLogger::instance().log( message, msg.level );
}

void addServerSendSink()
{
    auto serverSendSink = std::make_shared<ServerSendSink>();
    Logger::instance().addSink( serverSendSink );
}

} //namespace MR
