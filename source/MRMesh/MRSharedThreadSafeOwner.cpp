#include "MRSharedThreadSafeOwner.h"
#include "MRAABBTree.h"
#include "MRAABBTreePolyline.h"
#include "MRAABBTreePoints.h"
#include "MRDipole.h"
#include "MRHeapBytes.h"
#include "MRPch/MRTBB.h"
#include <cassert>

namespace MR
{

struct TaskGroup : tbb::task_group
{};

template<typename T>
void SharedThreadSafeOwner<T>::reset()
{
    assert( !construction_.load() ); // one thread constructs the object, and this thread resets it
    obj_.store( {} );
}

template<typename T>
void SharedThreadSafeOwner<T>::update( const std::function<void(T&)> & updater )
{
    assert( !construction_.load() ); // one thread constructs the object, and this thread resets it
    auto myPtr = obj_.exchange( {} );
    if ( !myPtr )
        return;
    assert( myPtr.use_count() >= 1 );
    if ( myPtr.use_count() > 1 )
        myPtr.reset( new T( *myPtr ) );
    assert( myPtr.use_count() == 1 );
    updater( const_cast<T&>( *myPtr ) );
    obj_.exchange( std::move( myPtr ) );
}

template<typename T>
const T & SharedThreadSafeOwner<T>::getOrCreate( const std::function<T()> & creator )
{
    /// if many parallel threads call this function simultaneously, they will join one task_group
    /// and will cooperatively construct owned object
    for (;;)
    {
        //if ( obj_ ) // fast path to avoid increasing shared pointers when everything is ready
        //    return *obj_;
        auto myPtr = obj_.load();
        if ( myPtr )
            return *myPtr;
        assert( creator );
        std::shared_ptr<TaskGroup> construction;
        bool firstConstructor = construction_.compare_exchange_strong( construction, std::make_shared<TaskGroup>() );
        construction = construction_.load();
        assert( construction );

        myPtr = obj_.load();
        if ( myPtr ) // already constructed while we setup construction
        {
            if ( firstConstructor )
                construction_.store( {} );
            return *myPtr;
        }

        if ( firstConstructor )
        {
            // we do not want this thread while inside creator steal outside piece of work 
            // and call this function recursively, and stopping forever (because creator() never returns)
            // in construction->wait()
            tbb::this_task_arena::isolate( [&]
            {
               construction->run( [&]
               {
                   obj_.store( std::make_shared<const T>( creator() ) );
                   assert( construction == construction_.load() );
                   construction_.store( {} );
               } );
            } );
        }
        construction->wait();
        // several iterations are necessary only if the object was reset immediately after construction
    }
}

template<typename T>
size_t SharedThreadSafeOwner<T>::heapBytes() const
{
    auto myPtr = obj_.load();
    return MR::heapBytes( myPtr );
}

template class SharedThreadSafeOwner<AABBTree>;
template class SharedThreadSafeOwner<AABBTreePolyline2>;
template class SharedThreadSafeOwner<AABBTreePolyline3>;
template class SharedThreadSafeOwner<AABBTreePoints>;
template class SharedThreadSafeOwner<Dipoles>;

} //namespace MR
