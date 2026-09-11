#pragma once

#include "MRPch/MRBindingMacros.h"
#include "MRMeshFwd.h"

#include <iterator>
#include <memory>
#include <vector>

namespace MR
{

/// read-only access to the children of an Object, as returned by Object::constChildren()
///
/// Object stores its children as std::shared_ptr<Object>, and a const Object cannot hand
/// them out directly without either dropping constness or copying the whole vector.
/// This class solves that: it is a lightweight non-owning view on the object's children,
/// which produces std::shared_ptr<const Object> on the fly. Typical use:
///
///     for ( const auto & child : obj.constChildren() )
///         processRecursively( *child );
///
/// It is also indexable, so a plain loop works too:
///
///     for ( size_t i = 0; i < obj.constChildren().size(); ++i )
///         process( *obj.constChildren()[i] );
///
/// The view refers to the object's own storage, so use it as a temporary and do not store
/// it: it is invalidated by anything that adds, removes or reorders the object's children,
/// exactly like an iterator of the underlying vector.
class ConstChildren
{
public:
    using Storage = std::vector<std::shared_ptr<Object>>;

    /// dereferences to std::shared_ptr<const Object>, constructed on each dereference
    class MR_BIND_IGNORE_PY Iterator
    {
    public:
        /// dereference produces a prvalue, so the C++17 category can only be input,
        /// but the iterator is multi-pass and models std::forward_iterator
        using iterator_concept = std::forward_iterator_tag;
        using iterator_category = std::input_iterator_tag;
        using value_type = std::shared_ptr<const Object>;
        using difference_type = std::ptrdiff_t;

        Iterator() {}
        explicit Iterator( Storage::const_iterator it ) : it_( it ) {}

        [[nodiscard]] value_type operator *() const { return *it_; }
        Iterator & operator ++() { ++it_; return *this; }
        Iterator operator ++( int ) { auto res = *this; ++it_; return res; }
        [[nodiscard]] bool operator ==( const Iterator & ) const = default;

    private:
        Storage::const_iterator it_{};
    };

    explicit ConstChildren( const Storage & children ) : children_( children ) {}

    [[nodiscard]] MR_BIND_IGNORE_PY Iterator begin() const { return Iterator( children_.begin() ); }
    [[nodiscard]] MR_BIND_IGNORE_PY Iterator end() const { return Iterator( children_.end() ); }

    [[nodiscard]] bool empty() const { return children_.empty(); }
    [[nodiscard]] size_t size() const { return children_.size(); }
    [[nodiscard]] std::shared_ptr<const Object> operator []( size_t i ) const { return children_[i]; }

private:
    const Storage & children_;
};

} //namespace MR
