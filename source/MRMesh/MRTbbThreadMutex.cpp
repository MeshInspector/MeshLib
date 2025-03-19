#include "MRTbbThreadMutex.h"

namespace MR
{

TbbThreadMutex::TbbThreadMutex( std::thread::id id )
    : id_( id )
{
}

TbbThreadMutex::LockGuard::LockGuard( TbbThreadMutex& mutex )
    : mutex_( mutex )
{
}

TbbThreadMutex::LockGuard::~LockGuard()
{
    mutex_.lockFlag_.clear();
}

std::optional<TbbThreadMutex::LockGuard> TbbThreadMutex::tryLock()
{
    if ( std::this_thread::get_id() == id_ && !lockFlag_.test_and_set() )
        return LockGuard { *this };
    return {};
}

} // namespace MR
