#include "MRFeatureObject.h"
#include "MRMatrix3Decompose.h"
#include "MRObjectDimensionsEnum.h"
#include "MRSceneColors.h"
#include "MRSceneSettings.h"
#include "MRSerializer.h"

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

    root["PointSize"] = pointSize_;
    root["LineWidth"] = lineWidth_;
    root["SubPointSize"] = subPointSize_;
    root["SubLineWidth"] = subLineWidth_;

    root["MainAlpha"] = mainFeatureAlpha_;
    root["SubAlphaPoints"] = subAlphaPoints_;
    root["SubAlphaLines"] = subAlphaLines_;
    root["SubAlphaMesh"] = subAlphaMesh_;

    for ( std::size_t i = 0; i < std::size_t( DimensionsVisualizePropertyType::_count ); i++ )
    {
        const auto enumValue = DimensionsVisualizePropertyType( i );
        if ( supportsVisualizeProperty( enumValue ) )
            root["DimensionVisibility"][toString( enumValue ).data()] = getVisualizePropertyMask( enumValue ).value();
    }
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

    if ( const auto& json = root["PointSize"]; json.isDouble() )
        pointSize_ = json.asFloat();
    if ( const auto& json = root["LineWidth"]; json.isDouble() )
        lineWidth_ = json.asFloat();
    if ( const auto& json = root["SubPointSize"]; json.isDouble() )
        subPointSize_ = json.asFloat();
    if ( const auto& json = root["SubLineWidth"]; json.isDouble() )
        subLineWidth_ = json.asFloat();

    if ( const auto& json = root["MainAlpha"]; json.isDouble() )
        mainFeatureAlpha_ = json.asFloat();
    if ( const auto& json = root["SubAlphaPoints"]; json.isDouble() )
        subAlphaPoints_ = json.asFloat();
    if ( const auto& json = root["SubAlphaLines"]; json.isDouble() )
        subAlphaLines_ = json.asFloat();
    if ( const auto& json = root["SubAlphaMesh"]; json.isDouble() )
        subAlphaMesh_ = json.asFloat();

    for ( std::size_t i = 0; i < std::size_t( DimensionsVisualizePropertyType::_count ); i++ )
    {
        const auto enumValue = DimensionsVisualizePropertyType( i );
        if ( supportsVisualizeProperty( enumValue ) )
            if ( const auto& json = root["DimensionVisibility"][toString( enumValue ).data()]; json.isUInt() )
                setVisualizePropertyMask( enumValue, ViewportMask( json.asUInt() ) );
    }

    // only default xf value serialize now.
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

float FeatureObject::getPointSize() const
{
    return pointSize_;
}

float FeatureObject::getLineWidth() const
{
    return lineWidth_;
}

void FeatureObject::setPointSize( float pointSize )
{
    if ( pointSize_ != pointSize )
    {
        pointSize_ = pointSize;
        needRedraw_ = true;
    }
}

void FeatureObject::setLineWidth( float lineWidth )
{
    if ( lineWidth_ != lineWidth )
    {
        lineWidth_ = lineWidth;
        needRedraw_ = true;
    }
}

float FeatureObject::getSubfeaturePointSize() const
{
    return subPointSize_;
}

float FeatureObject::getSubfeatureLineWidth() const
{
    return subLineWidth_;
}

void FeatureObject::setSubfeaturePointSize( float pointSize )
{
    if ( subPointSize_ != pointSize )
    {
        subPointSize_ = pointSize;
        needRedraw_ = true;
    }
}

void FeatureObject::setSubfeatureLineWidth( float lineWidth )
{
    if ( subLineWidth_ != lineWidth )
    {
        subLineWidth_ = lineWidth;
        needRedraw_ = true;
    }
}

float FeatureObject::getMainFeatureAlpha() const
{
    return mainFeatureAlpha_;
}

float FeatureObject::getSubfeatureAlphaPoints() const
{
    return subAlphaPoints_;
}

float FeatureObject::getSubfeatureAlphaLines() const
{
    return subAlphaLines_;
}

float FeatureObject::getSubfeatureAlphaMesh() const
{
    return subAlphaMesh_;
}

void FeatureObject::setMainFeatureAlpha( float alpha )
{
    if ( mainFeatureAlpha_ != alpha )
    {
        mainFeatureAlpha_ = alpha;
        needRedraw_ = true;
    }
}

void FeatureObject::setSubfeatureAlphaPoints( float alpha )
{
    if ( subAlphaPoints_ != alpha )
    {
        subAlphaPoints_ = alpha;
        needRedraw_ = true;
    }
}

void FeatureObject::setSubfeatureAlphaLines( float alpha )
{
    if ( subAlphaLines_ != alpha )
    {
        subAlphaLines_ = alpha;
        needRedraw_ = true;
    }
}

void FeatureObject::setSubfeatureAlphaMesh( float alpha )
{
    if ( subAlphaMesh_ != alpha )
    {
        subAlphaMesh_ = alpha;
        needRedraw_ = true;
    }
}

FeatureObject::FeatureObject( int numDimensions )
{
    setLocked( true );

    setFrontColor( SceneColors::get( SceneColors::SelectedFeatures ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedFeatures ), false );
    setBackColor( SceneColors::get( SceneColors::FeatureBackFaces ) );

    setDecorationsColor( SceneColors::get( SceneColors::UnselectedFeatureDecorations ), false );
    setDecorationsColor( SceneColors::get( SceneColors::SelectedFeatureDecorations ), true );

    setPointSize( SceneSettings::get( SceneSettings::FloatType::FeaturePointSize ) );
    setLineWidth( SceneSettings::get( SceneSettings::FloatType::FeatureLineWidth ) );
    setSubfeaturePointSize( SceneSettings::get( SceneSettings::FloatType::FeatureSubPointSize ) );
    setSubfeatureLineWidth( SceneSettings::get( SceneSettings::FloatType::FeatureSubLineWidth ) );

    if ( numDimensions == 0 )
        setMainFeatureAlpha( SceneSettings::get( SceneSettings::FloatType::FeaturePointsAlpha ) );
    else if ( numDimensions == 1 )
        setMainFeatureAlpha( SceneSettings::get( SceneSettings::FloatType::FeatureLinesAlpha ) );
    else
        setMainFeatureAlpha( SceneSettings::get( SceneSettings::FloatType::FeatureMeshAlpha ) );

    setSubfeatureAlphaPoints( SceneSettings::get( SceneSettings::FloatType::FeatureSubPointsAlpha ) );
    setSubfeatureAlphaLines( SceneSettings::get( SceneSettings::FloatType::FeatureSubLinesAlpha ) );
    setSubfeatureAlphaMesh( SceneSettings::get( SceneSettings::FloatType::FeatureSubMeshAlpha ) );
}

void FeatureObject::setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos )
{
    VisualObject::setAllVisualizeProperties_( properties, pos );
    setAllVisualizePropertiesForEnum<FeatureVisualizePropertyType>( properties, pos );
}

Vector3f FeatureObject::getBasePoint( ViewportId id /*= {} */ ) const
{
    return xf( id ).b;
}

}
