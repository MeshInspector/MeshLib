#include "MRSharedThreadSafeOwner.h"
#include "MRAABBTree.h"
#include "MRAABBTreePolyline.h"
#include "MRAABBTreePoints.h"
#include "MRDipole.h"
#include "MRHeapBytes.h"
#include "MRPch/MRSuppressWarning.h"
#include "MRPch/MRTBB.h"
#include <cassert>

#if _GLIBCXX_RELEASE >= 14
MR_SUPPRESS_WARNING_PUSH
// atomic_load, atomic_store, atomic_exchange, atomic_compare_exchange are deprecated in C++20
MR_SUPPRESS_WARNING( "-Wdeprecated-declarations", 4996 )
#endif

namespace MR
{

struct TaskGroup : tbb::task_group
{};

template<typename T>
void SharedThreadSafeOwner<T>::reset()
{
    assert( !construction_ ); // one thread constructs the object, and this thread resets it
    obj_.reset();
}

template<typename T>
void SharedThreadSafeOwner<T>::update( const std::function<void(T&)> & updater )
{
    assert( !construction_ ); // one thread constructs the object, and this thread updates it
    auto myPtr = atomic_exchange( &obj_, {} );
    if ( !myPtr )
        return;
    assert( myPtr.use_count() >= 1 );
    if ( myPtr.use_count() > 1 ) // create the copy if we was not the unique owner
        myPtr.reset( new T( *myPtr ) );
    assert( myPtr.use_count() == 1 );
    updater( *myPtr );
    assert( !construction_ ); // one thread constructs the object, and this thread updates it
    atomic_store( &obj_, std::move( myPtr ) );
}

template<typename T>
const T & SharedThreadSafeOwner<T>::getOrCreate( const std::function<T()> & creator )
{
    assert( creator );
    /// if many parallel threads call this function simultaneously, they will join one task_group
    /// and will cooperatively construct owned object
    for (;;)
    {
        if ( auto p = obj_.get() ) // fastest path avoiding increasing/decreasing reference counter when everything is ready
            return *p;
        if ( auto p = atomic_load( &obj_ ) ) // fast path, which insures that the thread does not use old register-cached value
            return *p;


        bool firstConstructor = false; // true only the thread starting the creation
        auto construction = atomic_load( &construction_ );
        if ( !construction )
        {
            firstConstructor = atomic_compare_exchange_strong( &construction_, &construction, std::make_shared<TaskGroup>() );
            if ( firstConstructor )
                construction = atomic_load( &construction_ );
            assert( construction );
        }

        if ( auto p = atomic_load( &obj_ ) ) // already constructed while we setup construction
        {
            if ( firstConstructor )
                atomic_store( &construction_, {} );
            return *p;
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
                   atomic_store( &obj_, std::make_shared<T>( creator() ) );
                   assert( construction == atomic_load( &construction_ ) );
                   atomic_store( &construction_, {} );
               } );
            } );
        }
        construction->wait();
    }
}

template<typename T>
size_t SharedThreadSafeOwner<T>::heapBytes() const
{
    auto myPtr = atomic_load( &obj_ );
    return MR::heapBytes( myPtr );
}

template class SharedThreadSafeOwner<AABBTree>;
template class SharedThreadSafeOwner<AABBTreePolyline2>;
template class SharedThreadSafeOwner<AABBTreePolyline3>;
template class SharedThreadSafeOwner<AABBTreePoints>;
template class SharedThreadSafeOwner<Dipoles>;

} //namespace MR

#if _GLIBCXX_RELEASE >= 14
MR_SUPPRESS_WARNING_POP
#endif
