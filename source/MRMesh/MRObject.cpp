#include "MRObject.h"
#include "MRObjectFactory.h"
#include "MRSerializer.h"
#include "MRStringConvert.h"
#include "MRHeapBytes.h"
#include "MRPch/MRJson.h"
#include "MRGTest.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( Object )

ObjectChildrenHolder::ObjectChildrenHolder( ObjectChildrenHolder && b ) noexcept 
    : children_( std::move( b.children_ ) )
    , bastards_( std::move( b.bastards_ ) )
{
    auto * thisObject = static_cast<Object*>( this );
    for ( const auto & child : children_ )
        if ( child )
            child->parent_ = thisObject;
    for ( const auto & wchild : bastards_ )
        if ( auto child = wchild.lock() )
            child->parent_ = thisObject;
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
    auto * thisObject = static_cast<Object*>( this );
    for ( const auto & child : children_ )
        if ( child )
            child->parent_ = thisObject;
    for ( const auto & wchild : bastards_ )
        if ( auto child = wchild.lock() )
            child->parent_ = thisObject;
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

size_t ObjectChildrenHolder::heapBytes() const
{
    auto res = MR::heapBytes( children_ ) + MR::heapBytes( bastards_ );
    for ( const auto & child : children_ )
        if ( child )
            res += heapBytes();
    return res;
}

std::shared_ptr<const Object> Object::find( const std::string_view & name ) const
{
    for ( const auto & child : children_ )
        if ( child->name() == name )
            return child;
    return {}; // not found among recognized children
}

void Object::setXf( const AffineXf3f& xf )
{
    if ( xf_ == xf )
        return;
    xf_ = xf; 
    xfChangedSignal();
    needRedraw_ = true;
}

AffineXf3f Object::worldXf() const
{
    auto xf = xf_;
    auto parent = parent_;
    while ( parent )
    {
        xf = parent->xf() * xf;
        parent = parent->parent();
    }
    return xf;
}

void Object::setWorldXf( const AffineXf3f& worldxf )
{
    setXf( xf_ * worldXf().inverse() * worldxf );
}

void Object::applyScale( float )
{
}

ViewportMask Object::globalVisibilityMask() const
{
    auto res = visibilityMask_;
    auto parent = parent_;
    while ( !res.empty() && parent )
    {
        res &= parent->visibilityMask_;
        parent = parent->parent();
    }
    return res;
}

void Object::setGlobalVisibilty( bool on, ViewportMask viewportMask /*= ViewportMask::any() */ )
{
    setVisible( on, viewportMask );
    if ( !on )
        return;

    auto parent = parent_;
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
    auto preParent = parent_;
    while ( preParent )
    {
        if ( preParent == ancestor )
            return true;
        preParent = preParent->parent();
    }
    return false;
}

bool Object::detachFromParent()
{
    if ( !parent_ )
        return false;
    return parent_->removeChild( this );
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
        children_.push_back( std::move( child ) );
    }
    else
    {
        // remove invalid children before adding new one
        std::erase_if( bastards_, [](const auto & b) { return !b.lock(); } );
        bastards_.push_back( std::move( child ) );
    }

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
    children_.insert( it1, std::move( newChild ) );
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
}

bool Object::select( bool on )
{
    if ( selected_ == on )
        return false;

    if ( ancillary_ && on )
        return false;

    selected_ = on;
    return true;
}

void Object::setAncillary( bool ancillary )
{
    if ( ancillary )
        select( false );
    ancillary_ = ancillary;
}

void Object::setVisible( bool on, ViewportMask viewportMask /*= ViewportMask::all() */ )
{
    if ( ( visibilityMask_ & viewportMask ) == ( on ? viewportMask : ViewportMask{} ) )
        return;

    needRedraw_ = true;

    if ( on ) 
        setVisibilityMask( visibilityMask_ | viewportMask ); 
    else 
        setVisibilityMask( visibilityMask_ & ~viewportMask );
}

Object::Object( const Object& other )
{
    name_ = other.name_;
    xf_ = other.xf_;
    visibilityMask_ = other.visibilityMask_;
    locked_ = other.locked_;
    selected_ = other.selected_;
    ancillary_ = other.ancillary_;
}

void Object::swapBase_( Object& other )
{
    std::swap( *this, other );
}

void Object::swapSignals_( Object& other )
{
    std::swap( xfChangedSignal, other.xfChangedSignal );
}

tl::expected<std::future<void>, std::string> Object::serializeModel_( const std::filesystem::path& ) const
{
    return {};
}

void Object::serializeFields_( Json::Value& root ) const
{
    root["Name"] = name_;
    root["Visibility"] = visibilityMask_.value();
    root["Selected"] = selected_;
    root["Locked"] = locked_;

    // xf
    serializeToJson( xf_, root["XF"] );

    // Type
    root["Type"].append( Object::TypeName() ); // will be appended in derived calls
}

tl::expected<void, std::string> Object::deserializeModel_( const std::filesystem::path& )
{
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
        deserializeFromJson( root["XF"], xf_ );
    if ( root["Locked"].isBool() )
        locked_ = root["Locked"].asBool();
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

tl::expected<std::vector<std::future<void>>, std::string> Object::serializeRecursive( const std::filesystem::path& path, Json::Value& root,
    int childId ) const
{
    std::error_code ec;
    if ( !std::filesystem::is_directory( path, ec ) )
        if ( !std::filesystem::create_directories( path, ec ) )
            return tl::make_unexpected( "Cannot create directories " + utf8string( path ) );

    std::vector<std::future<void>> res;

    auto replaceProhibitedChars = []( const std::string & s )
    {
        auto res = s;
        for ( auto & c : res )
            if ( c == '?' || c == '*' || c == '/' || c == '\\' )
                c = '_';
        return res;
    };

    // the key must be unique among all children of same parent
    std::string key = std::to_string( childId ) + "_" + replaceProhibitedChars( name_ );

    auto model = serializeModel_( path / key );
    if ( !model.has_value() )
        return tl::make_unexpected( model.error() );
    if ( model.value().valid() )
        res.push_back( std::move( model.value() ) );
    serializeFields_( root );
    
    root["Key"] = key;

    if ( !children_.empty() )
    {
        auto childrenPath = path / key;
        auto& childrenRoot = root["Children"];
        for ( int i = 0; i < children_.size(); ++i )
        {
            const auto& child = children_[i];
            if ( child->isAncillary() )
                continue; // consider ancillary_ objects as temporary, not requiring saving
            auto sub = child->serializeRecursive( childrenPath, childrenRoot[std::to_string( i )], i );
            if ( !sub.has_value() )
                return tl::make_unexpected( sub.error() );
            for ( auto & f : sub.value() )
            {
                assert( f.valid() );
                res.push_back( std::move( f ) );
            }
        }
    }
    return res;
}

tl::expected<void, std::string> Object::deserializeRecursive( const std::filesystem::path& path, const Json::Value& root )
{
    std::string key = root["Key"].isString() ? root["Key"].asString() : root["Name"].asString();

    auto res = deserializeModel_( path / key );
    if ( !res.has_value() )
        return res;

    deserializeFields_( root );

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

            auto childRes = childObj->deserializeRecursive( path / key, child );
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

Box3f Object::getWorldTreeBox( ViewportMask viewportMask ) const
{
    Box3f res = getWorldBox();
    for ( const auto & c : children_ )
        if ( c && !c->isAncillary() && c->isVisible( viewportMask ) )
            res.include( c->getWorldTreeBox() );
    return res;
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
