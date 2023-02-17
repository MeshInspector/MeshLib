#pragma once
#include "MRViewerFwd.h"

namespace MR
{
// singleton class to manage blocking sessions among different plugins
// used in RibbonMenu mostly
class MRVIEWER_CLASS BlockingStateSession
{
public:
    // starts new session incrementing timestamp
    // return false if other session is active
    MRVIEWER_API static bool startNewSession();
    // stops current session
    MRVIEWER_API static void stopCurrentSession();
    // return id of current session, 0 if there is no current session
    MRVIEWER_API static size_t getCurrentSessionId();
private:
    BlockingStateSession() = default;
    static BlockingStateSession& instance_();
    size_t sessionTimeStamp_{ 0 };
    size_t currentSessionId_{ 0 };
};
}