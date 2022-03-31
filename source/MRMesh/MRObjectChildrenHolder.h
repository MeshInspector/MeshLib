#pragma once

#include <algorithm>
#include <cctype>
#include <memory>
#include <vector>

namespace MR
{

// since every object stores a pointer on its parent,
// copying of this object is prohibited and moving is taken with care
// This is a CRTP base class.
template <typename T>
class ObjectChildrenHolder
{
public:
    ObjectChildrenHolder() = default;
    ObjectChildrenHolder( const ObjectChildrenHolder& ) = delete;
    ObjectChildrenHolder& operator = ( const ObjectChildrenHolder& ) = delete;
    ObjectChildrenHolder( ObjectChildrenHolder&& b ) noexcept : children_( std::move( b.children_ ) )
    {
        auto* thisObject = static_cast<T*>( this );
        for ( const auto& child : children_ )
            if ( child )
                child->parent_ = thisObject;
    }
    ObjectChildrenHolder& operator = ( ObjectChildrenHolder&& b ) noexcept
    {
        for ( const auto & child : children_ )
        if ( child )
            child->parent_ = nullptr;

        children_ = std::move( b.children_ );
        auto* thisObject = static_cast<T*>( this );
        for ( const auto & child : children_ )
            if ( child )
                child->parent_ = thisObject;
        return *this;
    }
protected:
    ~ObjectChildrenHolder()
    {
        for ( const auto& child : children_ )
        if ( child )
            child->parent_ = nullptr;
    }
public:
    // returns parent object in the tree
    [[nodiscard]] virtual T* parent() { return parent_; }
    [[nodiscard]] const T* parent() const { return const_cast<ObjectChildrenHolder*>( this )->parent(); }

    // return true if given object is ancestor of this one, false otherwise
    // If `respectLogicalParents` is true, respects overridden virtual `parent()`.
    [[nodiscard]] bool isAncestor( const T* ancestor, bool respectLogicalParents = true ) const
    {
        if ( !ancestor )
            return false;
        auto preParent = parent_;
        while ( preParent )
        {
            if ( preParent == ancestor )
                return true;
            preParent = respectLogicalParents ? preParent->parent() : preParent->parent_;
        }
        return false;
    }

    // removes this from its parent children list
    // returns false if it was already orphan
    virtual bool detachFromParent()
    {
        if ( !parent_ )
            return false;
        return parent_->removeChild( static_cast<T*>( this ) );
    }
    // an object can hold other sub-objects
    [[nodiscard]] const std::vector<std::shared_ptr<T>>& children() { return children_; }
    [[nodiscard]] const std::vector<std::shared_ptr<const T>>& children() const { return reinterpret_cast<const std::vector<std::shared_ptr<const T>>&>( children_ ); }
    // adds given object at the end of this children;
    // returns false if it was already child of this, of if given pointer is empty
    virtual bool addChild( std::shared_ptr<T> child )
    {
        if( !child )
            return false;

        if ( child.get() == this )
            return false;

        auto oldParent = child->parent_;
        if ( oldParent == this )
            return false;

        if ( isAncestor( child.get(), false ) )
            return false;

        if ( oldParent )
            oldParent->removeChild( child );

        child->parent_ = static_cast<T*>( this );
        children_.push_back( std::move( child ) );

        return true;
    }
    // adds given object in this children before existingChild;
    // if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
    // returns false if newChild is nullptr, or existingChild is not a child of this
    virtual bool addChildBefore( std::shared_ptr<T> newChild, const std::shared_ptr<T> & existingChild )
    {
        if( !newChild || newChild == existingChild )
            return false;

        if ( newChild.get() == this )
            return false;

        auto it1 = std::find( begin( children_ ), end( children_ ), existingChild );
        if ( it1 == end( children_ ) )
            return false;

        if ( isAncestor( newChild.get(), false ) )
            return false;

        auto oldParent = newChild->parent_;
        if ( oldParent == this )
        {
            auto it0 = std::find( begin( children_ ), end( children_ ), newChild );
            if ( it0 == end( children_ ) )
            {
                assert( false );
                return false;
            }
            if ( it0 + 1 < it1 )
                std::rotate( it0, it0 + 1, it1 );
            else if ( it1 < it0 )
                std::rotate( it1, it0, it0 + 1 );
            return true;
        }

        if ( oldParent )
            oldParent->removeChild( newChild );

        newChild->parent_ = static_cast<T*>( this );
        children_.insert( it1, std::move( newChild ) );
        return true;
    }
    // returns false if it was not child of this
    bool removeChild( const std::shared_ptr<T>& child ) { return removeChild( child.get() ); }

    virtual bool removeChild( T* child )
    {
        assert( child );
        if ( !child )
            return false;

        auto oldParent = child->parent_;
        if ( oldParent != this )
            return false;

        child->parent_ = nullptr;

        auto it = std::remove_if( children_.begin(), children_.end(), [child]( const std::shared_ptr<Object>& obj )
        {
            return obj.get() == child;
        } );
        assert( it != children_.end() );
        children_.erase( it, children_.end() );

        return true;
    }

    virtual void removeAllChildren()
    {
        for ( const auto & ch : children_ )
            ch->parent_ = nullptr;
        children_.clear();
    }

    /// sort children by name
    virtual void sortChildren()
    {
        std::sort( children_.begin(), children_.end(), [] ( const auto& a, const auto& b )
        {
            const auto& lhs = a->name();
            const auto& rhs = b->name();
            // used for case insensitive sorting
            const auto result = std::mismatch( lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend(),
                [] ( const unsigned char lhsc, const unsigned char rhsc )
            {
                return std::tolower( lhsc ) == std::tolower( rhsc );
            } );

            return result.second != rhs.cend() && ( result.first == lhs.cend() || std::tolower( *result.first ) < std::tolower( *result.second ) );
        } );
    }
private:
    T* parent_ = nullptr;
    std::vector<std::shared_ptr<T>> children_;
};

}
