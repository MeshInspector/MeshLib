#include "MRUniqueThreadSafeOwner.h"
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
UniqueThreadSafeOwner<T>::UniqueThreadSafeOwner() = default;

template<typename T>
UniqueThreadSafeOwner<T>::UniqueThreadSafeOwner( const UniqueThreadSafeOwner& b ) 
{ 
    assert( this != &b );
    // do not lock this since nobody can use it before the end of construction
    std::unique_lock lock( b.mutex_ );
    if ( b.obj_ )
        obj_.reset( new T( *b.obj_ ) );
}

template<typename T>
UniqueThreadSafeOwner<T>& UniqueThreadSafeOwner<T>::operator =( const UniqueThreadSafeOwner& b ) 
{
    if ( this != &b )
    {
        std::scoped_lock lock( mutex_, b.mutex_ );
        obj_.reset();
        if ( b.obj_ )
            obj_.reset( new T( *b.obj_ ) );
    }
    return *this; 
}

template<typename T>
UniqueThreadSafeOwner<T>::UniqueThreadSafeOwner( UniqueThreadSafeOwner&& b ) noexcept
{
    assert( this != &b );
    // do not lock this since nobody can use it before the end of construction
    std::unique_lock lock( b.mutex_ );
    obj_ = std::move( b.obj_ );
}

template<typename T>
UniqueThreadSafeOwner<T>& UniqueThreadSafeOwner<T>::operator =( UniqueThreadSafeOwner&& b ) noexcept
{
    if ( this != &b )
    {
        std::scoped_lock lock( mutex_, b.mutex_ );
        obj_ = std::move( b.obj_ );
    }
    return *this;
}

template<typename T>
UniqueThreadSafeOwner<T>::~UniqueThreadSafeOwner() = default;

template<typename T>
void UniqueThreadSafeOwner<T>::reset()
{
    std::unique_lock lock( mutex_ );
    assert( !construction_ ); // one thread constructs the object, and this thread resets it
    obj_.reset();
}

template<typename T>
void UniqueThreadSafeOwner<T>::update( const std::function<void(T&)> & updater )
{
    std::unique_lock lock( mutex_ );
    assert( !construction_ ); // one thread constructs the object, and this thread updates it
    if ( obj_ )
        updater( *obj_ );
}

template<typename T>
T & UniqueThreadSafeOwner<T>::getOrCreate( const std::function<T()> & creator )
{
    /// if many parallel threads call this function simultaneously, they will join one task_group
    /// and will cooperatively construct owned object
    for (;;)
    {
        if ( obj_ ) // fast path to avoid locking when everything is ready
            return *obj_;
        assert( creator );
        bool firstConstructor = false;
        std::shared_ptr<TaskGroup> construction;
        {
            std::unique_lock lock( mutex_ );
            if ( obj_ ) // already constructed while we waited for lock
            {
                assert( !construction_ );
                return *obj_;
            }
            if ( !construction_ )
            {
                construction_ = std::make_unique<TaskGroup>();
                firstConstructor = true;
            }
            construction = construction_;
        }
        assert( construction );
        if ( firstConstructor )
        {
            // we do not want this thread while inside creator steal outside piece of work 
            // and call this function recursively, and stopping forever (because creator() never returns)
            // in construction->wait()
            tbb::this_task_arena::isolate( [&]
            {
               construction->run( [&]
               {
                   auto newObj = std::make_unique<T>( creator() );
                   std::unique_lock lock( mutex_ );
                   assert( construction == construction_ );
                   construction_.reset();
                   obj_ = std::move( newObj );
               } );
            } );
        }
        construction->wait();
        // several iterations are necessary only if the object was reset immediately after construction
    }
}

template<typename T>
size_t UniqueThreadSafeOwner<T>::heapBytes() const
{
    std::unique_lock lock( mutex_ );
    return MR::heapBytes( obj_ );
}

template class UniqueThreadSafeOwner<AABBTree>;
template class UniqueThreadSafeOwner<AABBTreePolyline2>;
template class UniqueThreadSafeOwner<AABBTreePolyline3>;
template class UniqueThreadSafeOwner<AABBTreePoints>;
template class UniqueThreadSafeOwner<Dipoles>;

} //namespace MR
