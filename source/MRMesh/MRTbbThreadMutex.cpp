#include "MRTbbThreadMutex.h"

namespace MR
{

TbbThreadMutex::TbbThreadMutex( std::thread::id id )
    : id_( id )
{
}

TbbThreadMutex::LockGuard::LockGuard( LockGuard&& other )
{
    mutex_ = other.mutex_;
    other.mutex_ = nullptr;
}

TbbThreadMutex::LockGuard::~LockGuard()
{
    if ( mutex_ )
        mutex_->lockFlag_.clear();
}

std::optional<TbbThreadMutex::LockGuard> TbbThreadMutex::tryLock()
{
    if ( std::this_thread::get_id() == id_ && !lockFlag_.test_and_set() )
        return LockGuard { *this };
    return {};
}

} // namespace MR
