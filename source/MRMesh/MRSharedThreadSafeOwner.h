#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"
#include <functional>
#include <memory>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

/// A group of these objects owns collectively one T-instance,
/// allowing access to stored instance from parallel threads;
/// if one object updates its T-instance, then it makes a copy of T-instance first and become the only object in new group
template<typename T>
class MR_BIND_IGNORE SharedThreadSafeOwner
{
public:
    /// stops owning T-instance
    MRMESH_API void reset();

    /// returns the currently owned instance, the pointer becomes invalid after reset() or update()
    [[nodiscard]] const T* get() { return obj_.get(); }

    /// returns the currently owned instance
    [[nodiscard]] std::shared_ptr<const T> getPtr() { return obj_; }

    /// returns the currently owned instance (if any),
    /// otherwise calls (creator) to create new owned instance, which is returned;
    /// if many threads call this simultaneously, then they can collectively participate in the construction
    MRMESH_API const T & getOrCreate( const std::function<T()> & creator );

    /// if the object owns some T-instance, then updater function is applied to it;
    /// get() and getPtr() return nullptr foe other threads during update()
    MRMESH_API void update( const std::function<void(T&)> & updater );

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

private:
    /// collectively owned T-instance
    std::shared_ptr<T> obj_;

    /// not-null during creation of owned instance only
    std::shared_ptr<TbbTaskArenaAndGroup> construction_;
};

/// \}

} // namespace MR
