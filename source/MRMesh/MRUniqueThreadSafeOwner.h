#pragma once

#include "MRMeshFwd.h"
#include <functional>
#include <mutex>
#include <memory>

namespace MR
{

struct TaskGroup;

/// \addtogroup AABBTreeGroup
/// \{

/// This class is base class for unique thread safe owning of some objects, for example AABBTree
/// classes derived from this one should have function like getOrCreate
template<typename T>
class UniqueThreadSafeOwner
{
public:
    MRMESH_API UniqueThreadSafeOwner();
    MRMESH_API UniqueThreadSafeOwner( const UniqueThreadSafeOwner& );
    MRMESH_API UniqueThreadSafeOwner& operator =( const UniqueThreadSafeOwner& );
    MRMESH_API UniqueThreadSafeOwner( UniqueThreadSafeOwner&& b ) noexcept;
    MRMESH_API UniqueThreadSafeOwner& operator =( UniqueThreadSafeOwner&& b ) noexcept;
    MRMESH_API ~UniqueThreadSafeOwner();

    /// deletes owned object
    MRMESH_API void reset();
    /// returns existing owned object and does not create new one
    T * get() { return obj_.get(); }
    /// returns existing owned object or creates new one using creator function
    MRMESH_API T & getOrCreate( const std::function<T()> & creator );
    /// calls given updater for the owned object (if any)
    MRMESH_API void update( const std::function<void(T&)> & updater );
    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

private:
    mutable std::mutex mutex_;
    std::unique_ptr<T> obj_;
    /// not-null during creation of owned object only
    std::shared_ptr<TaskGroup> construction_;
};

/// \}

} // namespace MR
