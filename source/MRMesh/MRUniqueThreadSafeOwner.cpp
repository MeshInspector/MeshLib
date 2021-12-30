#include "MRUniqueThreadSafeOwner.h"
#include "MRAABBTree.h"
#include "MRAABBTreePolyline.h"
#include "MRAABBTreePoints.h"
#include <tbb/task_arena.h>
#include <cassert>

namespace MR
{

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
    obj_.reset();
}

template<typename T>
const T & UniqueThreadSafeOwner<T>::getOrCreate( const std::function<T()> & creator )
{
    if ( obj_ ) // fast path to avoid locking when everything is ready
        return *obj_;
    assert( creator );
    std::unique_lock lock( mutex_ );
    if ( !obj_ )
    {
        // we do not want this thread while inside creator steal outside piece of work 
        // and call UniqueThreadSafeOwner<T>::getOrCreate recursively
        tbb::this_task_arena::isolate( [&]
        {
            obj_ = std::make_unique<T>( creator() );
        } );
    }
    return *obj_;
}

template class UniqueThreadSafeOwner<AABBTree>;
template class UniqueThreadSafeOwner<AABBTreePolyline2>;
template class UniqueThreadSafeOwner<AABBTreePolyline3>;
template class UniqueThreadSafeOwner<AABBTreePoints>;

} //namespace MR
