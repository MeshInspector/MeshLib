#include "MRFeatureObject.h"
#include "MRMatrix3Decompose.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRSerializer.h"

#include "json/value.h"

namespace MR
{

bool FeatureObject::supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const
{
    return VisualObject::supportsVisualizeProperty( type ) || type.tryGet<FeatureVisualizePropertyType>().has_value();
}

AllVisualizeProperties FeatureObject::getAllVisualizeProperties() const
{
    AllVisualizeProperties ret = VisualObject::getAllVisualizeProperties();
    getAllVisualizePropertiesForEnum<FeatureVisualizePropertyType>( ret );
    return ret;
}

const ViewportMask& FeatureObject::getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const
{
    if ( auto value = type.tryGet<FeatureVisualizePropertyType>() )
    {
        switch ( *value )
        {
            case FeatureVisualizePropertyType::Subfeatures:
                return subfeatureVisibility_;
            case FeatureVisualizePropertyType::DetailsOnNameTag:
                return detailsOnNameTag_;
            case FeatureVisualizePropertyType::_count: break; // MSVC warns if this is missing, despite `[[maybe_unused]]` on the `_count`.
        }
        assert( false && "Invalid enum." );
        return visibilityMask_;
    }
    else
    {
        return VisualObject::getVisualizePropertyMask( type );
    }
}

void FeatureObject::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    // append base type
    root["Type"].append( VisualObject::TypeName() );

    root["SubfeatureVisibility"] = subfeatureVisibility_.value();
    root["DetailsOnNameTag"] = detailsOnNameTag_.value();

    serializeToJson( Vector4f( decorationsColor_[0].get() ), root["DecorationsColorUnselected"] );
    serializeToJson( Vector4f( decorationsColor_[1].get() ), root["DecorationsColorSelected"] );
}

void FeatureObject::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );

    if ( const auto& subfeatureVisibilityJson = root["SubfeatureVisibility"]; subfeatureVisibilityJson.isUInt() )
        subfeatureVisibility_ = ViewportMask( subfeatureVisibilityJson.asUInt() );

    if ( const auto& detailsOnNameTagJson = root["DetailsOnNameTag"]; detailsOnNameTagJson.isUInt() )
        detailsOnNameTag_ = ViewportMask( detailsOnNameTagJson.asUInt() );

    Vector4f color;
    deserializeFromJson( root["DecorationsColorUnselected"], color ); decorationsColor_[0] = Color( color );
    deserializeFromJson( root["DecorationsColorSelected"], color ); decorationsColor_[1] = Color( color );

    // only default xf value serialyze now.
    decomposeMatrix3( xf().A, r_.get(), s_.get() );
}

std::optional<Vector3f> FeatureObject::getNormal( const Vector3f& point ) const
{
    return projectPoint( point ).normal;
}

void FeatureObject::setXf( const AffineXf3f& xf, ViewportId id )
{
    if ( VisualObject::xf( id ) == xf )
        return;
    decomposeMatrix3( xf.A, r_[id], s_[id] );
    VisualObject::setXf( xf, id );
}

void FeatureObject::resetXf( ViewportId id )
{
    r_.reset( id );
    s_.reset( id );

    VisualObject::resetXf( id );
}

const Color& FeatureObject::getDecorationsColor( bool selected, ViewportId viewportId, bool* isDef ) const
{
    // Calling the getter in case it's overridden.
    return getDecorationsColorForAllViewports( selected ).get( viewportId, isDef );
}

void FeatureObject::setDecorationsColor( const Color& color, bool selected, ViewportId viewportId )
{
    auto& target = decorationsColor_[selected];
    if ( target.get( viewportId ) != color )
        target.set( color, viewportId );
}

const ViewportProperty<Color>& FeatureObject::getDecorationsColorForAllViewports( bool selected ) const
{
    return decorationsColor_[selected];
}

void FeatureObject::setDecorationsColorForAllViewports( ViewportProperty<Color> val, bool selected )
{
    decorationsColor_[selected] = std::move( val );
}

FeatureObject::FeatureObject()
{
    setDecorationsColor( SceneColors::get( SceneColors::UnselectedFeatureDecorations ), false );
    setDecorationsColor( SceneColors::get( SceneColors::SelectedFeatureDecorations ), true );
}

void FeatureObject::setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos )
{
    VisualObject::setAllVisualizeProperties_( properties, pos );
    setAllVisualizePropertiesForEnum<FeatureVisualizePropertyType>( properties, pos );
}

}
