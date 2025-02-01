#pragma once

#include "MRMeshFwd.h"
#include <atomic>
#include <functional>
#include <memory>

namespace MR
{

struct TaskGroup;

/// \addtogroup AABBTreeGroup
/// \{

/// This class is base class for unique thread safe owning of some objects, for example AABBTree
/// classes derived from this one should have function like getOrCreate
template<typename T>
class SharedThreadSafeOwner
{
public:
    /// deletes owned object
    MRMESH_API void reset();

    /// returns existing owned object and does not create new one
    const T* get() { return obj_.load().get(); }

    /// returns existing owned object and does not create new one
    std::shared_ptr<const T> getPtr() { return obj_.load(); }

    /// returns existing owned object or creates new one using creator function
    MRMESH_API const T & getOrCreate( const std::function<T()> & creator );

    /// calls given updater for the owned object (if any)
    MRMESH_API void update( const std::function<void(T&)> & updater );

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

private:
    std::atomic<std::shared_ptr<const T>> obj_;

    /// not-null during creation of owned object only
    std::atomic<std::shared_ptr<TaskGroup>> construction_;
};

/// \}

} // namespace MR
