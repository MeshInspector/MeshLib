#include "MRObject.h"
#include "MRObjectFactory.h"
#include "MRSerializer.h"
#include "MRStringConvert.h"
#include "MRPch/MRJson.h"
#include <filesystem>
#include "MRGTest.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( Object )

std::shared_ptr<const Object> Object::find( const std::string_view & name ) const
{
    for ( const auto & child : children() )
        if ( child->name() == name )
            return child;
    return {}; // not found
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
    auto obj = parent();
    while ( obj )
    {
        xf = obj->xf() * xf;
        obj = obj->parent();
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

bool Object::globalVisibilty( ViewportMask viewportMask /*= ViewportMask::any() */ ) const
{
    bool visible = isVisible( viewportMask );
    auto obj = parent();
    while ( visible && obj )
    {
        visible = obj->isVisible( viewportMask );
        obj = obj->parent();
    }
    return visible;
}

void Object::setGlobalVisibilty( bool on, ViewportMask viewportMask /*= ViewportMask::any() */ )
{
    setVisible( on, viewportMask );
    if ( !on )
        return;

    auto obj = parent();
    while ( obj )
    {
        obj->setVisible( true, viewportMask );
        obj = obj->parent();
    }
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
    for ( const auto& child : children() )
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
    for ( const auto& child : children() )
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

    if ( !children().empty() )
    {
        auto childrenPath = path / key;
        auto& childrenRoot = root["Children"];
        for ( int i = 0; i < children().size(); ++i )
        {
            const auto& child = children()[i];
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
