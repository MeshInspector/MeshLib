#pragma once

#include "MRMeshFwd.h"
#include "MRVector4.h"
#include "MRAffineXf3.h"
#include "MRBox.h"
#include "MRBitSet.h"
#include "MRViewportProperty.h"
#include "MRProgressCallback.h"
#include <boost/signals2/signal.hpp>
#include <tl/expected.hpp>
#include <string>
#include <memory>
#include <vector>
#include <array>
#include <future>
#include <filesystem>

namespace Json
{
class Value;
}

namespace MR
{

/**
 * \defgroup DataModelGroup Data Model
 * \brief This chapter represents documentation about data models
 * \{
 */

/// This class exists to provide default copy and move operations on std::mutex
class MutexOwner
{
public:
    MutexOwner() noexcept = default;
    MutexOwner( const MutexOwner& ) noexcept{}
    MutexOwner( MutexOwner&& ) noexcept{}
    MutexOwner& operator =( const MutexOwner& ){return *this;};
    MutexOwner& operator =( MutexOwner&& ) noexcept{return *this;};
    std::mutex& getMutex(){return mutex_;}
private:
    std::mutex mutex_;
};


/// since every object stores a pointer on its parent,
/// copying of this object is prohibited and moving is taken with care
struct ObjectChildrenHolder
{
    ObjectChildrenHolder() = default;
    ObjectChildrenHolder( const ObjectChildrenHolder & ) = delete;
    ObjectChildrenHolder & operator = ( const ObjectChildrenHolder & ) = delete;
    MRMESH_API ObjectChildrenHolder( ObjectChildrenHolder && ) noexcept;
    MRMESH_API ObjectChildrenHolder & operator = ( ObjectChildrenHolder && ) noexcept;
    MRMESH_API ~ObjectChildrenHolder();

    /// returns the amount of memory this object occupies on heap,
    /// including the memory of all recognized children
    [[nodiscard]] size_t heapBytes() const;

protected:
    Object * parent_ = nullptr;
    std::vector< std::shared_ptr< Object > > children_; /// recognized ones
    std::vector< std::weak_ptr< Object > > bastards_; /// unrecognized children to hide from the pubic
};

/// named object in the data model
class MRMESH_CLASS Object : public ObjectChildrenHolder
{
public:
    Object() = default;
    Object( Object && ) noexcept = default;
    Object & operator = ( Object && ) noexcept = default;
    virtual ~Object() = default;

    // return name of subtype for serialization purposes
    constexpr static const char* TypeName() noexcept { return "Object"; }
    virtual const char* typeName() const { return TypeName(); }

    template <typename T>
    T * asType() { return dynamic_cast<T*>( this ); }
    template <typename T>
    const T * asType() const { return dynamic_cast<const T*>( this ); }

    const std::string & name() const { return name_; }
    virtual void setName( std::string name ) { name_ = std::move( name ); }

    /// finds a direct child by name
    MRMESH_API std::shared_ptr<const Object> find( const std::string_view & name ) const;
    std::shared_ptr<Object> find( const std::string_view & name ) { return std::const_pointer_cast<Object>( const_cast<const Object*>( this )->find( name ) ); }

    /// finds a direct child by name and type
    template <typename T>
    std::shared_ptr<const T> find( const std::string_view & name ) const;
    template <typename T>
    std::shared_ptr<T> find( const std::string_view & name ) { return std::const_pointer_cast<T>( const_cast<const Object*>( this )->find<T>( name ) ); }

    /// this space to parent space transformation (to world space if no parent) for default viewport
    const AffineXf3f & xf() const { return xf_.get(); }
    MRMESH_API virtual void setXf( const AffineXf3f& xf );
    /// this space to parent space transformation for specific viewport
    const AffineXf3f & xf( ViewportId id ) const { return xf_.get( id ); }
    MRMESH_API virtual void setXf( ViewportId id, const AffineXf3f& xf );

    /// this space to world space transformation for default viewport
    MRMESH_API AffineXf3f worldXf() const;
    MRMESH_API void setWorldXf( const AffineXf3f& xf );
    /// this space to world space transformation for specific viewport
    MRMESH_API AffineXf3f worldXf( ViewportId id ) const;
    MRMESH_API void setWorldXf( ViewportId id, const AffineXf3f& xf );

    /// scale object size (all point positions)
    MRMESH_API virtual void applyScale( float scaleFactor );

    /// returns all viewports where this object is visible together with all its parents
    MRMESH_API ViewportMask globalVisibilityMask() const;
    /// returns true if this object is visible together with all its parents in any of given viewports
    bool globalVisibilty( ViewportMask viewportMask = ViewportMask::any() ) const { return !( globalVisibilityMask() & viewportMask ).empty(); }
    /// if true sets all predecessors visible, otherwise sets this object invisible
    MRMESH_API void setGlobalVisibilty( bool on, ViewportMask viewportMask = ViewportMask::any() );

    /// object properties lock for UI
    const bool isLocked() const { return locked_; }
    virtual void setLocked( bool on ) { locked_ = on; }

    /// returns parent object in the tree
    const Object * parent() const { return parent_; }
    Object * parent() { return parent_; }

    /// return true if given object is ancestor of this one, false otherwise
    MRMESH_API bool isAncestor( const Object* ancestor ) const;

    /// removes this from its parent children list
    /// returns false if it was already orphan
    MRMESH_API virtual bool detachFromParent();
    /// an object can hold other sub-objects
    const std::vector<std::shared_ptr<Object>>& children() { return children_; }
    const std::vector<std::shared_ptr<const Object>>& children() const { return reinterpret_cast<const std::vector< std::shared_ptr< const Object > > &>( children_ ); }
    /// adds given object at the end of children (recognized or not);
    /// returns false if it was already child of this, of if given pointer is empty
    MRMESH_API virtual bool addChild( std::shared_ptr<Object> child, bool recognizedChild = true );
    /// adds given object in the recognized children before existingChild;
    /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
    /// returns false if newChild is nullptr, or existingChild is not a child of this
    MRMESH_API virtual bool addChildBefore( std::shared_ptr<Object> newChild, const std::shared_ptr<Object> & existingChild );
    /// returns false if it was not child of this
    bool removeChild( const std::shared_ptr<Object>& child ) { return removeChild( child.get() ); }
    MRMESH_API virtual bool removeChild( Object* child );
    /// detaches all recognized children from this, keeping all unrecognized ones
    MRMESH_API virtual void removeAllChildren();
    /// sort recognized children by name
    MRMESH_API void sortChildren();

    /// selects the object, returns true if value changed, otherwise returns false
    MRMESH_API virtual bool select( bool on );
    bool isSelected() const { return selected_; }

    /// ancillary object is an object hidden (in scene menu) from a regular user
    /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
    MRMESH_API virtual void setAncillary( bool ancillary );
    bool isAncillary() const { return ancillary_; }

    /// sets the object visible in the viewports specified by the mask (by default in all viewports)
    MRMESH_API void setVisible( bool on, ViewportMask viewportMask = ViewportMask::all() );
    /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
    bool isVisible( ViewportMask viewportMask = ViewportMask::any() ) const { return !( visibilityMask_ & viewportMask ).empty(); }
    /// specifies object visibility as bitmask of viewports
    virtual void setVisibilityMask( ViewportMask viewportMask ) { visibilityMask_ = viewportMask; }
    /// gets object visibility as bitmask of viewports
    ViewportMask visibilityMask() const { return visibilityMask_; }

    /// this method virtual because others data model types could have dirty flags or something 
    virtual bool getRedrawFlag( ViewportMask ) const { return needRedraw_; }
    void resetRedrawFlag() const { needRedraw_ = false; }

    /// clones all tree of this object (except ancillary and unrecognized children)
    MRMESH_API std::shared_ptr<Object> cloneTree() const;
    /// clones current object only, without parent and/or children
    MRMESH_API virtual std::shared_ptr<Object> clone() const;
    /// clones all tree of this object (except ancillary and unrecognied children)
    /// clones only pointers to mesh, points or voxels
    MRMESH_API std::shared_ptr<Object> shallowCloneTree() const;
    /// clones current object only, without parent and/or children
    /// clones only pointers to mesh, points or voxels
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const;

    /// return several info lines that can better describe object in the UI
    MRMESH_API virtual std::vector<std::string> getInfoLines() const;
    /// return human readable name of subclass
    virtual std::string getClassName() const { return "Object"; }

    /// creates futures that save this object subtree:
    ///   models in the folder by given path and
    ///   fields in given JSON
    /// \param childId is its ordinal number within the parent
    tl::expected<std::vector<std::future<void>>, std::string> serializeRecursive( const std::filesystem::path& path,
        Json::Value& root, int childId ) const;

    /// loads subtree into this Object
    ///   models from the folder by given path and
    ///   fields from given JSON
    tl::expected<void, std::string> deserializeRecursive( const std::filesystem::path& path, const Json::Value& root,
        ProgressCallback progressCb = {}, int* objCounter = nullptr );

    /// swaps this object with other
    /// note: do not swap object signals, so listeners will get notifications from swapped object
    /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
    MRMESH_API void swap( Object& other );

    /// returns bounding box of this object in world coordinates
    virtual Box3f getWorldBox() const { return {}; } ///empty box
    /// returns bounding box of this object and all children visible in given viewports in world coordinates
    MRMESH_API Box3f getWorldTreeBox( ViewportMask viewportMask = ViewportMask::any() ) const;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const;

    /// signal about xf changing, triggered in setXf and setWorldXf,  it is called for children too
    using XfChangedSignal = boost::signals2::signal<void() >;
    XfChangedSignal worldXfChangedSignal;
protected:
    struct ProtectedStruct{ explicit ProtectedStruct() = default; };
public:
    /// \note this ctor is public only for std::make_shared used inside clone()
    Object( ProtectedStruct, const Object& obj ) : Object( obj ) {}

protected:
    /// user should not be able to call copy implicitly, use clone() function instead
    MRMESH_API Object( const Object& obj );

    /// swaps whole object (signals too)
    MRMESH_API virtual void swapBase_( Object& other );
    /// swaps signals, used in `swap` function to return back signals after `swapBase_`
    /// \note pls call Parent::swapSignals_ first when overriding this function
    MRMESH_API virtual void swapSignals_( Object& other );

    /// Creates future to save object model (e.g. mesh) in given file
    /// path is full filename without extension
    MRMESH_API virtual tl::expected<std::future<void>, std::string> serializeModel_( const std::filesystem::path& path ) const;

    /// Write parameters to given Json::Value,
    /// \note if you override this method, please call Base::serializeFields_(root) in the beginning
    MRMESH_API virtual void serializeFields_( Json::Value& root ) const;

    /// Reads model from file
    MRMESH_API virtual tl::expected<void, std::string> deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb = {} );

    /// Reads parameters from json value
    /// \note if you override this method, please call Base::deserializeFields_(root) in the beginning
    MRMESH_API virtual void deserializeFields_( const Json::Value& root );

    std::string name_;
    ViewportProperty<AffineXf3f> xf_;
    mutable MutexOwner readCacheMutex_;
    ViewportMask visibilityMask_ = ViewportMask::all();
    bool locked_ = false;
    bool selected_{ false };
    bool ancillary_{ false };
    mutable bool needRedraw_{false};

    void propagateWorldXfChangedSignal_();
};

template <typename T>
std::shared_ptr<const T> Object::find( const std::string_view & name ) const
{
    for ( const auto & child : children_ )
        if ( child->name() == name )
            if ( auto res = std::dynamic_pointer_cast<T>( child ) )
                return res;
    return {}; // not found
}

/// \}

} ///namespace MR
