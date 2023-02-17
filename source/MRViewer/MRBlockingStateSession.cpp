#include "MRBlockingStateSession.h"

namespace MR
{

bool BlockingStateSession::startNewSession()
{
    auto& instance = instance_();
    if ( instance.currentSessionId_ != 0 )
        return false;
    instance.sessionTimeStamp_++;
    instance.currentSessionId_ = instance.sessionTimeStamp_;
    return true;
}

void BlockingStateSession::stopCurrentSession()
{
    instance_().currentSessionId_ = 0;
}

size_t BlockingStateSession::getCurrentSessionId()
{
    return instance_().currentSessionId_;
}

BlockingStateSession& BlockingStateSession::instance_()
{
    static BlockingStateSession instance;
    return instance;
}

}