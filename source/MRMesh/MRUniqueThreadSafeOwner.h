#pragma once

#include "MRMeshFwd.h"
#include <functional>
#include <mutex>
#include <memory>

namespace MR
{

// This class is base class for unique thread safe owning of some objects, for example AABBTree
// classes derived from this one should have function like getOrCreate
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

    // deletes owned object
    MRMESH_API void reset();
    // returns existing owned object or creates new one using creator function
    MRMESH_API const T & getOrCreate( const std::function<T()> & creator );

protected:
    mutable std::mutex mutex_;
    std::unique_ptr<T> obj_;
};

} //namespace MR
