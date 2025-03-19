#include "MRThreadSemaphore.h"

namespace MR
{

ThreadSemaphore::ThreadSemaphore( std::thread::id id )
    : id_( id )
    , semaphore_( 0 )
{
}

ThreadSemaphore::Lock::Lock( ThreadSemaphore& semaphore )
    : id_( semaphore.id_ )
    , semaphore_( semaphore.semaphore_ )
    , acquired_( std::this_thread::get_id() == id_ && semaphore_.fetch_add( 1, std::memory_order_relaxed ) == 0 )
{
}

ThreadSemaphore::Lock::~Lock()
{
    if ( std::this_thread::get_id() == id_ )
        semaphore_.fetch_sub( 1, std::memory_order_relaxed );
}

ThreadSemaphore::Lock ThreadSemaphore::acquire()
{
    return { *this };
}

} // namespace MR
