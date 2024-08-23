#pragma once

#include "MRVoxelsFwd.h"

#include "MRMesh/MRProgressCallback.h"
#include <openvdb/util/NullInterrupter.h>
#include <algorithm>
#include <thread>

namespace MR
{

// This class implements OpenVdb interrupter interface and provides ability to use MR::ProgressCallback in some OpenVdb operations
struct ProgressInterrupter : openvdb::util::NullInterrupter
{
    ProgressInterrupter( ProgressCallback cb )  : cb_{ cb }
        , progressThreadId_{ std::this_thread::get_id() } {}
    virtual bool wasInterrupted( int percent = -1 ) override
    {
        // OpenVdb uses worker threads from pool, in addition to the caller thread.
        // It is assumed that the callback is periodically called in the caller thread.
        if ( cb_ && progressThreadId_ == std::this_thread::get_id() )
            wasInterrupted_ = !cb_( float( std::clamp( percent, 0, 100 ) ) / 100.0f );
        return wasInterrupted_;
    }
    bool getWasInterrupted() const { return wasInterrupted_; }

private:
    bool wasInterrupted_{ false };
    ProgressCallback cb_;
    std::thread::id progressThreadId_;
};

} //namespace MR
