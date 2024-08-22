#include "MRObjectImGuiLabel.h"

#include "MRMesh/MRObjectFactory.h"
#include "MRPch/MRJson.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectImGuiLabel )

ObjectImGuiLabel::ObjectImGuiLabel() {}

std::shared_ptr<Object> ObjectImGuiLabel::clone() const
{
    return std::make_shared<ObjectImGuiLabel>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> ObjectImGuiLabel::shallowClone() const
{
    return ObjectImGuiLabel::clone();
}

ObjectImGuiLabel::ObjectImGuiLabel( ProtectedStruct, const ObjectImGuiLabel& obj )
    : ObjectImGuiLabel( obj )
{}

const std::string& ObjectImGuiLabel::getLabel() const
{
    return labelText_;
}

void ObjectImGuiLabel::setLabel( std::string value )
{
    labelText_ = std::move( value );
}

void ObjectImGuiLabel::swapBase_( Object& other )
{
    if ( auto ptr = other.asType<ObjectImGuiLabel>() )
        std::swap( *this, *ptr );
    else
        assert( false );
}

void ObjectImGuiLabel::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );
    root["Type"].append( TypeName() );

    root["LabelText"] = labelText_;
}

void ObjectImGuiLabel::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );

    if ( const auto& json = root["LabelText"]; json.isString() )
        labelText_ = json.asString();
}

void ObjectImGuiLabel::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}


}
