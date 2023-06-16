#include "MRUniqueThreadSafeOwner.h"
#include "MRAABBTree.h"
#include "MRAABBTreePolyline.h"
#include "MRAABBTreePoints.h"
#include "MRHeapBytes.h"
#include "MRPch/MRTBB.h"
#include <cassert>

namespace MR
{

struct CollaborativeCreation : tbb::collaborative_once_flag
{};

template<typename T>
UniqueThreadSafeOwner<T>::UniqueThreadSafeOwner() = default;

template<typename T>
UniqueThreadSafeOwner<T>::UniqueThreadSafeOwner( const UniqueThreadSafeOwner& b ) 
{ 
    assert( this != &b );
    // do not lock this since nobody can use it before the end of collaborativeCreation
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
    // do not lock this since nobody can use it before the end of collaborativeCreation
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
    assert( !collaborativeCreation_ ); // one thread constructs the object, and this thread resets it
    obj_.reset();
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
        std::shared_ptr<CollaborativeCreation> collaborativeCreation;
        {
            std::unique_lock lock( mutex_ );
            if ( obj_ ) // already constructed while we waited for lock
            {
                assert( !collaborativeCreation_ );
                return *obj_;
            }
            if ( !collaborativeCreation_ )
                collaborativeCreation_ = std::make_unique<CollaborativeCreation>();
            collaborativeCreation = collaborativeCreation_;
        }
        assert( collaborativeCreation );
        tbb::collaborative_call_once( *collaborativeCreation, [&] {
            auto newObj = std::make_unique<T>( creator() );
            std::unique_lock lock( mutex_ );
            assert( collaborativeCreation == collaborativeCreation_ );
            collaborativeCreation_.reset();
            obj_ = std::move( newObj );
        } );
        // several iterations are necessary only if the object was reset immediately after collaborativeCreation
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

} //namespace MR
