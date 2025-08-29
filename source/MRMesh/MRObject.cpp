#include "MRObject.h"
#include "MRObjectFactory.h"
#include "MRObjectTagEventDispatcher.h"
#include "MRSerializer.h"
#include "MRStringConvert.h"
#include "MRHeapBytes.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRSpdlog.h"
#include "MRGTest.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( Object )

ObjectChildrenHolder::ObjectChildrenHolder( ObjectChildrenHolder && b ) noexcept
    : children_( std::move( b.children_ ) )
    , bastards_( std::move( b.bastards_ ) )
{
    for ( const auto & child : children_ )
        if ( child )
            child->parent_ = this;
    for ( const auto & wchild : bastards_ )
        if ( auto child = wchild.lock() )
            child->parent_ = this;
}

ObjectChildrenHolder & ObjectChildrenHolder::operator = ( ObjectChildrenHolder && b ) noexcept
{
    for ( const auto & child : children_ )
        if ( child )
            child->parent_ = nullptr;
    for ( const auto & wchild : bastards_ )
        if ( auto child = wchild.lock() )
            child->parent_ = nullptr;

    children_ = std::move( b.children_ );
    bastards_ = std::move( b.bastards_ );
    for ( const auto & child : children_ )
        if ( child )
            child->parent_ = this;
    for ( const auto & wchild : bastards_ )
        if ( auto child = wchild.lock() )
            child->parent_ = this;
    return * this;
}

ObjectChildrenHolder::~ObjectChildrenHolder()
{
    for ( const auto & child : children_ )
        if ( child )
            child->parent_ = nullptr;
    for ( const auto & wchild : bastards_ )
        if ( auto child = wchild.lock() )
            child->parent_ = nullptr;
}

std::shared_ptr<Object> ObjectChildrenHolder::getSharedPtr() const
{
    if ( !parent_ )
        return {};
    for ( const auto& child : parent_->children_ )
        if ( child.get() == this )
            return child;
    return {};
}

size_t ObjectChildrenHolder::heapBytes() const
{
    auto res = MR::heapBytes( children_ ) + MR::heapBytes( bastards_ );
    for ( const auto & child : children_ )
        if ( child )
            res += child->heapBytes();
    return res;
}

std::shared_ptr<const Object> Object::find( const std::string_view & name ) const
{
    for ( const auto & child : children_ )
        if ( child->name() == name )
            return child;
    return {}; // not found among recognized children
}

void Object::setXf( const AffineXf3f& xf, ViewportId id )
{
    if ( xf_.get( id ) == xf )
        return;
    if ( xf.A.det() == 0 )
    {
        assert( false && "Object transform is degenerate" );
        spdlog::warn( "Object transform is degenerate" );
        return;
    }
    xf_.set( xf, id );
    sendWorldXfChangedSignal_();
    needRedraw_ = true;
}

void Object::resetXf( ViewportId id )
{
    if ( !xf_.reset( id ) )
        return;
    sendWorldXfChangedSignal_();
    needRedraw_ = true;
}

AffineXf3f Object::worldXf( ViewportId id, bool * isDef ) const
{
    auto xf = xf_.get( id, isDef );
    auto parent = this->parent();
    while ( parent )
    {
        bool parentDef = true;
        xf = parent->xf( id, &parentDef ) * xf;
        if ( isDef )
            *isDef = *isDef && parentDef;
        parent = parent->parent();
    }
    return xf;
}

void Object::setWorldXf( const AffineXf3f& worldxf, ViewportId id )
{
    setXf( xf_.get( id ) * worldXf( id ).inverse() * worldxf );
}

void Object::applyScale( float )
{
}

ViewportMask Object::globalVisibilityMask() const
{
    auto res = visibilityMask();
    auto parent = this->parent();
    while ( !res.empty() && parent )
    {
        res &= parent->visibilityMask();
        parent = parent->parent();
    }
    return res;
}

void Object::setGlobalVisibility( bool on, ViewportMask viewportMask /*= ViewportMask::any() */ )
{
    setVisible( on, viewportMask );
    if ( !on )
        return;

    auto parent = this->parent();
    while ( parent )
    {
        parent->setVisible( true, viewportMask );
        parent = parent->parent();
    }
}

bool Object::isAncestor( const Object* ancestor ) const
{
    if ( !ancestor )
        return false;
    auto preParent = this->parent();
    while ( preParent )
    {
        if ( preParent == ancestor )
            return true;
        preParent = preParent->parent();
    }
    return false;
}

Object* Object::findCommonAncestor( Object& other )
{
    // Some common cases first.
    if ( this == &other )
        return this;
    if ( parent() == other.parent() )
        return parent();

    // Visits all parents of `cur`. Writes the number of them into `depth`.
    // Returns the topmost parent.
    auto visitParents = []( Object& object, int &depth ) -> Object&
    {
        Object* cur = &object;
        while ( true )
        {
            if ( auto parent = cur->parent() )
            {
                cur = parent;
                depth++;
            }
            else
            {
                return *cur;
            }
        }
    };

    int depthA = 0;
    int depthB = 0;
    if ( &visitParents( *this, depthA ) != &visitParents( other, depthB ) )
        return nullptr; // No common ancestor.

    Object* curA = this;
    Object* curB = &other;

    while ( depthA > depthB )
    {
        curA = curA->parent();
        depthA--;
    }
    while ( depthA < depthB )
    {
        curB = curB->parent();
        depthB--;
    }

    while ( curA != curB )
    {
        curA = curA->parent();
        curB = curB->parent();
    }

    return curA;
}

bool Object::detachFromParent()
{
    if ( !parent_ )
        return false;
    return parent()->removeChild( this );
}

bool Object::addChild( std::shared_ptr<Object> child, bool recognizedChild )
{
    if( !child )
        return false;

    if ( child.get() == this )
        return false;

    auto oldParent = child->parent();
    if ( oldParent == this )
        return false;

    if ( isAncestor( child.get() ) )
        return false;

    if ( oldParent )
        oldParent->removeChild( child );

    child->parent_ = this;
    if ( recognizedChild )
    {
        children_.push_back( child );
    }
    else
    {
        // remove invalid children before adding new one
        std::erase_if( bastards_, [](const auto & b) { return !b.lock(); } );
        bastards_.push_back( child );
    }
    child->sendWorldXfChangedSignal_();
    needRedraw_ = true;
    return true;
}

bool Object::addChildBefore( std::shared_ptr<Object> newChild, const std::shared_ptr<Object> & existingChild )
{
    if( !newChild || newChild == existingChild )
        return false;

    if ( newChild.get() == this )
        return false;

    auto it1 = std::find( begin( children_ ), end( children_ ), existingChild );
    if ( it1 == end( children_ ) )
        return false;

    if ( isAncestor( newChild.get() ) )
        return false;

    auto oldParent = newChild->parent();
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

    newChild->parent_ = this;
    children_.insert( it1, newChild );
    newChild->sendWorldXfChangedSignal_();
    needRedraw_ = true;
    return true;
}

bool Object::removeChild( Object* child )
{
    assert( child );
    if ( !child )
        return false;

    auto oldParent = child->parent();
    if ( oldParent != this )
        return false;

    child->parent_ = nullptr;
    needRedraw_ = true;

    auto it = std::remove_if( children_.begin(), children_.end(), [child]( const std::shared_ptr<Object>& obj )
    {
        return !obj || obj.get() == child;
    } );
    if ( it != children_.end() )
    {
        children_.erase( it, children_.end() );
        return true;
    }

    auto bit = std::remove_if( bastards_.begin(), bastards_.end(), [child]( const std::weak_ptr<Object>& wobj )
    {
        auto obj = wobj.lock();
        return !obj || obj.get() == child;
    } );
    assert( bit != bastards_.end() );
    bastards_.erase( bit, bastards_.end() );

    return true;
}

void Object::removeAllChildren()
{
    for ( const auto & ch : children_ )
        ch->parent_ = nullptr;
    children_.clear();
    needRedraw_ = true;
}

void Object::sortChildren()
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
    needRedraw_ = true;
}

bool Object::select( bool on )
{
    if ( selected_ == on )
        return false;

    if ( ancillary_ && on )
        return false;

    needRedraw_ = true;
    selected_ = on;
    return true;
}

void Object::setAncillary( bool ancillary )
{
    if ( ancillary )
        select( false );
    needRedraw_ = true;
    ancillary_ = ancillary;
}

void Object::setVisible( bool on, ViewportMask viewportMask /*= ViewportMask::all() */ )
{
    if ( ( visibilityMask_ & viewportMask ) == ( on ? viewportMask : ViewportMask{} ) )
        return;

    if ( on )
        setVisibilityMask( visibilityMask_ | viewportMask );
    else
        setVisibilityMask( visibilityMask_ & ~viewportMask );
}

void Object::setVisibilityMask( ViewportMask viewportMask )
{
    if ( visibilityMask_ == viewportMask )
        return;

    needRedraw_ = true;
    visibilityMask_ = viewportMask;
}

void Object::swapBase_( Object& other )
{
    std::swap( *this, other );
}

void Object::swapSignals_( Object& other )
{
    std::swap( worldXfChangedSignal, other.worldXfChangedSignal );
}

Expected<std::future<Expected<void>>> Object::serializeModel_( const std::filesystem::path& ) const
{
    return {};
}

void Object::serializeFields_( Json::Value& root ) const
{
    root["Name"] = name_;
    root["Visibility"] = visibilityMask().value();
    root["Selected"] = selected_;
    root["Locked"] = locked_;
    root["ParentLocked"] = parentLocked_;

    // xf
    serializeToJson( xf_.get(), root["XF"] );

    // Type
    root["Type"].append( Object::TypeName() ); // will be appended in derived calls

    // tags
    auto& tagsJson = root["Tags"] = Json::arrayValue;
    for ( const auto& tag : tags_ )
        tagsJson.append( tag );
}

Expected<void> Object::deserializeModel_( const std::filesystem::path&, ProgressCallback progressCb )
{
    if ( progressCb && !progressCb( 1.f ) )
        return unexpectedOperationCanceled();
    return{};
}

void Object::deserializeFields_( const Json::Value& root )
{
    if ( root["Name"].isString() )
        name_ = root["Name"].asString();
    if ( root["Visibility"].isUInt() )
    {
        unsigned mask = root["Visibility"].asUInt();
        if ( mask == 1 )
            mask = ~unsigned( 0 );
        visibilityMask_ = ViewportMask{ mask };
    }
    if ( root["Selected"].isBool() )
        selected_ = root["Selected"].asBool();
    if ( !root["XF"].isNull() )
        deserializeFromJson( root["XF"], xf_.get() );
    if ( root["Locked"].isBool() )
        locked_ = root["Locked"].asBool();
    if ( const auto& json = root["ParentLocked"]; json.isBool() )
        parentLocked_ = json.asBool();
    if ( const auto& tagsJson = root["Tags"]; tagsJson.isArray() )
        for ( const auto& tagJson : tagsJson )
            if ( tagJson.isString() )
                tags_.emplace( tagJson.asString() );
}

void Object::sendWorldXfChangedSignal_()
{
    std::stack<Object*> buf;
    buf.push( this );

    while ( !buf.empty() )
    {
        auto obj = buf.top();
        obj->onWorldXfChanged_();
        buf.pop();

        for ( auto& child : obj->children_ )
            buf.push( child.get() );
    }
}

void Object::onWorldXfChanged_()
{
    worldXfChangedSignal();
}

std::shared_ptr<Object> Object::cloneTree() const
{
    std::shared_ptr<Object> res = clone();
    for ( const auto& child : children_ )
        if ( !child->isAncillary() )
            res->addChild( child->cloneTree() );
    return res;
}

std::shared_ptr<Object> Object::clone() const
{
    return std::make_shared<Object>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> Object::shallowCloneTree() const
{
    std::shared_ptr<Object> res = shallowClone();
    for ( const auto& child : children_ )
        if ( !child->isAncillary() )
            res->addChild( child->shallowCloneTree() );
    return res;
}

std::shared_ptr<Object> Object::shallowClone() const
{
    return clone();
}

std::vector<std::string> Object::getInfoLines() const
{
    std::vector<std::string> res;

    res.push_back( "class: " + className() );
    res.push_back( "mem: " + bytesString( heapBytes() ) );
    res.push_back( fmt::format( "tags: {}", tags_.size() ) );
    for ( const auto& tag : tags_ )
        res.push_back( fmt::format( "· {}", tag ) );
    return res;
}

Expected<std::vector<std::future<Expected<void>>>> Object::serializeRecursive( const std::filesystem::path& path, Json::Value& root, int childId ) const
{
    std::error_code ec;
    if ( !std::filesystem::is_directory( path, ec ) )
        if ( !std::filesystem::create_directories( path, ec ) )
            return unexpected( "Cannot create directories " + utf8string( path ) );

    std::vector<std::future<Expected<void>>> res;

    // the key must be unique among all children of same parent
    std::string key = std::to_string( childId ) + "_" + replaceProhibitedChars( name_ );

    auto model = serializeModel_( path / pathFromUtf8( key ) );
    if ( !model.has_value() )
        return unexpected( model.error() );
    if ( model.value().valid() )
        res.push_back( std::move( model.value() ) );
    serializeFields_( root );

    root["Key"] = key;

    if ( !children_.empty() )
    {
        auto childrenPath = path / pathFromUtf8( key );
        auto& childrenRoot = root["Children"];
        for ( int i = 0; i < children_.size(); ++i )
        {
            const auto& child = children_[i];
            if ( child->isAncillary() )
                continue; // consider ancillary_ objects as temporary, not requiring saving
            auto sub = child->serializeRecursive( childrenPath, childrenRoot[std::to_string( i )], i );
            if ( !sub.has_value() )
                return unexpected( sub.error() );
            for ( auto & f : sub.value() )
            {
                assert( f.valid() );
                res.push_back( std::move( f ) );
            }
        }
    }
    return res;
}

Expected<void> Object::deserializeRecursive( const std::filesystem::path& path, const Json::Value& root,
        ProgressCallback progressCb, int* objCounter )
{
    std::string key = root["Key"].isString() ? root["Key"].asString() : root["Name"].asString();

    auto res = deserializeModel_( path / pathFromUtf8( key ), progressCb );
    if ( !res.has_value() )
        return res;

    deserializeFields_( root );
    if ( objCounter )
        ++( *objCounter );

    if (!root["Children"].isNull())
    {
        // split keys by type to sort numeric
        std::vector<long> orderedLongChildKeys; // all that can be converted to Long type
        std::vector<std::string> orderedStringChildKeys; // others
        for ( const std::string& childKey : root["Children"].getMemberNames() )
        {
            char* p_end;
            long childKeyAsLong = std::strtol( childKey.c_str(), &p_end, 10 ); // check if key can be converted to Long
            if ( *p_end )  // stoi failed
                orderedStringChildKeys.push_back( childKey );
            else
                orderedLongChildKeys.push_back( childKeyAsLong );
        }
        std::sort( orderedLongChildKeys.begin(), orderedLongChildKeys.end() );

        // join keys: after sorted numeric add all string keys
        std::vector<std::string> orderedKeys;
        orderedKeys.reserve( orderedLongChildKeys.size() + orderedStringChildKeys.size() );
        for ( const long& k : orderedLongChildKeys )
            orderedKeys.push_back( std::to_string( k ) );
        orderedKeys.insert( orderedKeys.end(),
                            std::make_move_iterator( orderedStringChildKeys.begin() ),
                            std::make_move_iterator( orderedStringChildKeys.end() ) );

        for ( const std::string& child_key : orderedKeys )
        {
            if (!root["Children"].isMember( child_key ))
            {
                assert( false );
                continue;
            }
            const auto& child = root["Children"][child_key];
            if (child.isNull())
                continue;

            auto typeTreeSize = child["Type"].size();
            std::shared_ptr<Object> childObj;
            for (int i = typeTreeSize -1;i>=0;--i)
            {
                const auto& type = child["Type"][unsigned(i)];
                if ( type.isString() )
                    childObj = createObject( type.asString() );
                if ( childObj )
                    break;
            }
            if ( !childObj )
                continue;

            auto childRes = childObj->deserializeRecursive( path / pathFromUtf8( key ), child, progressCb, objCounter );
            if ( !childRes.has_value() )
                return childRes;
            addChild( childObj );
        }
    }
    return {};
}

void Object::swap( Object& other )
{
    swapBase_( other );
    // swap signals second time to return in place
    swapSignals_( other );
}

Box3f Object::getWorldTreeBox( ViewportId id ) const
{
    Box3f res = getWorldBox( id );
    for ( const auto & c : children_ )
        if ( c && !c->isAncillary() && c->isVisible( id ) )
            res.include( c->getWorldTreeBox( id ) );
    return res;
}

bool Object::addTag( std::string tag )
{
    const auto [it, inserted] = tags_.emplace( std::move( tag ) );
    if ( inserted )
        ObjectTagEventDispatcher::instance().tagAddedSignal( this, *it );
    return inserted;
}

bool Object::removeTag( const std::string& tag )
{
    const auto present = bool( tags_.erase( tag ) );
    if ( present )
        ObjectTagEventDispatcher::instance().tagRemovedSignal( this, tag );
    return present;
}

size_t Object::heapBytes() const
{
    return ObjectChildrenHolder::heapBytes()
        + name_.capacity();
}

TEST( MRMesh, DataModelRemoveChild )
{
    auto child2 = std::make_shared<Object>();
    Object root;
    {
        EXPECT_EQ( root.children().size(), 0 );

        auto child1 = std::make_shared<Object>();
        EXPECT_TRUE( root.addChild( child1 ) );
        EXPECT_FALSE( root.addChild( child1 ) );
        EXPECT_EQ( &root, child1->parent() );
        EXPECT_EQ( root.children().size(), 1 );


        EXPECT_TRUE( child1->addChild( child2 ) );
        EXPECT_FALSE( child1->addChild( child2 ) );
        EXPECT_EQ( child1.get(), child2->parent() );
        EXPECT_EQ( child1->children().size(), 1 );

        EXPECT_TRUE( root.removeChild( child1 ) );
        EXPECT_FALSE( root.removeChild( child1 ) );
        EXPECT_EQ( nullptr, child1->parent() );
        EXPECT_EQ( root.children().size(), 0 );
    }

    auto parent = child2->parent();
    EXPECT_EQ( parent, nullptr );
}
} //namespace MR
