#include "MRVisualObject.h"
#include "MRObjectFactory.h"
#include "MRSerializer.h"
#include "MRSceneColors.h"
#include "MRTimer.h"
#include "MRHeapBytes.h"
#include "MRStringConvert.h"
#include "MRExpected.h"
#include "MRParallelFor.h"
#include "MRSceneSettings.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRSuppressWarning.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( VisualObject )

VisualObject::VisualObject()
{
    useDefaultScenePropertiesOnDeserialization_ = SceneSettings::get( SceneSettings::BoolType::UseDefaultScenePropertiesOnDeserialization );
    setDefaultSceneProperties_();
}

bool VisualObject::supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const
{
    return type.tryGet<VisualizeMaskType>().has_value();
}

void VisualObject::setVisualizeProperty( bool value, AnyVisualizeMaskEnum type, ViewportMask viewportMask )
{
    auto res = getVisualizePropertyMask( type );
    if ( value )
        res |= viewportMask;
    else
        res &= ~viewportMask;

    setVisualizePropertyMask( type, res );
}

void VisualObject::setVisualizePropertyMask( AnyVisualizeMaskEnum type, ViewportMask viewportMask )
{
    auto& mask = getVisualizePropertyMask_( type );
    if ( mask == viewportMask )
        return;
    mask = viewportMask;
    needRedraw_ = true;
}

bool VisualObject::getVisualizeProperty( AnyVisualizeMaskEnum type, ViewportMask viewportMask ) const
{
    return !( getVisualizePropertyMask( type ) & viewportMask ).empty();
}

void VisualObject::toggleVisualizeProperty( AnyVisualizeMaskEnum type, ViewportMask viewportMask )
{
    setVisualizePropertyMask( type, getVisualizePropertyMask( type ) ^ viewportMask );
}

void VisualObject::setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos )
{
    setAllVisualizePropertiesForEnum<VisualizeMaskType>( properties, pos );
}

AllVisualizeProperties VisualObject::getAllVisualizeProperties() const
{
    AllVisualizeProperties res;
    getAllVisualizePropertiesForEnum<VisualizeMaskType>( res );
    return res;
}

ViewportMask VisualObject::globalClippedByPlaneMask() const
{
    // do not access clipByPlane_ directly, to allow subclasses to override the behavior
    auto res = getVisualizePropertyMask( VisualizeMaskType::ClippedByPlane );
    auto parent = this->parent();
    while ( parent )
    {
        if ( auto visParent = dynamic_cast<const VisualObject*>( parent ) )
            res |= visParent->getVisualizePropertyMask( VisualizeMaskType::ClippedByPlane );
        parent = parent->parent();
    }
    return res;
}

void VisualObject::setGlobalClippedByPlane( bool on, ViewportMask viewportMask )
{
    setVisualizeProperty( on, VisualizeMaskType::ClippedByPlane, viewportMask );
    if ( on )
        return;

    auto parent = this->parent();
    while ( parent )
    {
        if ( auto visParent = dynamic_cast<VisualObject*>( parent ) )
            visParent->setVisualizeProperty( on, VisualizeMaskType::ClippedByPlane, viewportMask );
        parent = parent->parent();
    }
}

const Color& VisualObject::getFrontColor( bool selected /*= true */, ViewportId viewportId /*= {} */ ) const
{
    // Calling the getter in case it's overridden.
    return getFrontColorsForAllViewports( selected ).get( viewportId );
}

void VisualObject::setFrontColor( const Color& color, bool selected, ViewportId viewportId )
{
    if ( selected && selectedColor_.get( viewportId ) != color)
    {
        selectedColor_.set( color, viewportId );
    }
    else if ( !selected && unselectedColor_.get( viewportId ) != color )
    {
        unselectedColor_.set( color, viewportId );
    }
    needRedraw_ = true;
}

const ViewportProperty<Color>& VisualObject::getFrontColorsForAllViewports( bool selected ) const
{
    return selected ? selectedColor_ : unselectedColor_;
}

void VisualObject::setFrontColorsForAllViewports( ViewportProperty<Color> val, bool selected )
{
    selected ? selectedColor_ = std::move( val ) : unselectedColor_ = std::move( val );
    needRedraw_ = true;
}

const ViewportProperty<Color>& VisualObject::getBackColorsForAllViewports() const
{
    return backFacesColor_;
}

void VisualObject::setBackColorsForAllViewports( ViewportProperty<Color> val )
{
    backFacesColor_ = std::move( val );
    needRedraw_ = true;
}

const Color& VisualObject::getBackColor( ViewportId viewportId ) const
{
    // Calling the getter in case it's overridden.
    return getBackColorsForAllViewports().get( viewportId );
}

void VisualObject::setBackColor( const Color& color, ViewportId viewportId )
{
    if ( backFacesColor_.get( viewportId ) == color )
        return;
    backFacesColor_.set( color, viewportId );
    needRedraw_ = true;
}

const uint8_t& VisualObject::getGlobalAlpha( ViewportId viewportId /*= {} */ ) const
{
    // Calling the getter in case it's overridden.
    return getGlobalAlphaForAllViewports().get( viewportId );
}

void VisualObject::setGlobalAlpha( uint8_t alpha, ViewportId viewportId /*= {} */ )
{
    globalAlpha_.set( alpha, viewportId );
    needRedraw_ = true;
}

const ViewportProperty<uint8_t>& VisualObject::getGlobalAlphaForAllViewports() const
{
    return globalAlpha_;
}

void VisualObject::setGlobalAlphaForAllViewports( ViewportProperty<uint8_t> val )
{
    globalAlpha_ = std::move( val );
    needRedraw_ = true;
}

void VisualObject::setDirtyFlags( uint32_t mask, bool )
{
    if ( mask & DIRTY_FACE ) // first to also activate all flags due to DIRTY_POSITION later
        mask |= DIRTY_POSITION | DIRTY_UV | DIRTY_VERTS_COLORMAP;
    if ( mask & DIRTY_POSITION )
        mask |= DIRTY_RENDER_NORMALS | DIRTY_BOUNDING_BOX | DIRTY_BORDER_LINES | DIRTY_EDGES_SELECTION;
    // DIRTY_POSITION because we use corner rendering and need to update render verts
    // DIRTY_UV because we need to update UV coordinates

    dirty_ |= mask;

    needRedraw_ = true; // this is needed to differ dirty render object and dirty scene
}

void VisualObject::resetDirty() const
{
    // Bounding box and normals (all caches) is cleared only if it was recounted
    dirty_ &= DIRTY_CACHES;
}

void VisualObject::resetDirtyExceptMask( uint32_t mask ) const
{
    // Bounding box and normals (all caches) is cleared only if it was recounted
    dirty_ &= ( DIRTY_CACHES | mask );
}

Box3f VisualObject::getBoundingBox() const
{
    if ( dirty_ & DIRTY_BOUNDING_BOX )
    {
        boundingBoxCache_ = computeBoundingBox_();
        dirty_ &= ~DIRTY_BOUNDING_BOX;
    }
    return boundingBoxCache_;
}

void VisualObject::setPickable( bool on, ViewportMask viewportMask /*= ViewportMask::all() */ )
{
    if ( on )
        pickable_ |= viewportMask;
    else
        pickable_ &= ~viewportMask;
}

void VisualObject::setColoringType( ColoringType coloringType )
{
    if ( coloringType == coloringType_ )
        return;
    coloringType_ = coloringType;
    switch ( coloringType )
    {
    case ColoringType::SolidColor:
        needRedraw_ = true;
        break;
    case ColoringType::PrimitivesColorMap:
        dirty_ |= DIRTY_PRIMITIVE_COLORMAP;
        break;
    case ColoringType::VertsColorMap:
        dirty_ |= DIRTY_VERTS_COLORMAP;
        break;
    default:
        break;
    }
}

std::shared_ptr<Object> VisualObject::clone() const
{
    return std::make_shared<VisualObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> VisualObject::shallowClone() const
{
    return clone();
}

bool VisualObject::render( const ModelRenderParams& params ) const
{
    setupRenderObject_();
    if ( !renderObj_ )
        return false;

    return renderObj_->render( params );
}

void VisualObject::renderForPicker( const ModelBaseRenderParams& params, unsigned id ) const
{
    setupRenderObject_();
    if ( !renderObj_ )
        return;

    renderObj_->renderPicker( params, id );
}

void VisualObject::renderUi( const UiRenderParams& params ) const
{
    setupRenderObject_();
    if ( !renderObj_ )
        return;

    renderObj_->renderUi( params );
}

void VisualObject::swapBase_( Object& other )
{
    if ( auto otherVis = other.asType<VisualObject>() )
        std::swap( *this, *otherVis );
    else
        assert( false );
}

ViewportMask& VisualObject::getVisualizePropertyMask_( AnyVisualizeMaskEnum type )
{
    return const_cast< ViewportMask& >( getVisualizePropertyMask( type ) );
}

const ViewportMask& VisualObject::getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const
{
    if ( auto value = type.tryGet<VisualizeMaskType>() )
    {
        switch ( *value )
        {
        case VisualizeMaskType::Visibility:
            (void)visibilityMask(); // Call this for the side effects, in case it's overridden. Can't return it directly, as it returns by value.
            return visibilityMask_;
        case VisualizeMaskType::InvertedNormals:
            return invertNormals_;
        case VisualizeMaskType::ClippedByPlane:
            return clipByPlane_;
        case VisualizeMaskType::Name:
            return showName_;
        case VisualizeMaskType::DepthTest:
            return depthTest_;
        case VisualizeMaskType::_count: break; // MSVC warns if this is missing, despite `[[maybe_unused]]` on the `_count`.
        }
        assert( false && "Invalid enum." );
        return visibilityMask_;
    }
    else
    {
        assert( false && "Unknown `AnyVisualizeMaskEnum`." );
        return visibilityMask_;
    }
}

void VisualObject::serializeFields_( Json::Value& root ) const
{
    Object::serializeFields_( root );
    root["InvertNormals"] = !invertNormals_.empty();

    auto writeColors = [&root]( const char * fieldName, const Color& val )
    {
        auto& colors = root["Colors"]["Faces"][fieldName];
        serializeToJson( Vector4f( val ), colors["Diffuse"] );// To support old version
    };

    writeColors( "SelectedMode", selectedColor_.get() );
    writeColors( "UnselectedMode", unselectedColor_.get() );
    writeColors( "BackFaces", backFacesColor_.get() );

    root["Colors"]["GlobalAlpha"] = globalAlpha_.get();

    root["ShowName"] = showName_.value();

    // append base type
    root["Type"].append( VisualObject::TypeName() );

    root["UseDefaultSceneProperties"] = useDefaultScenePropertiesOnDeserialization_;
}

void VisualObject::deserializeFields_( const Json::Value& root )
{
    Object::deserializeFields_( root );

    if ( root["InvertNormals"].isBool() ) // Support old versions
        invertNormals_ = root["InvertNormals"].asBool() ? ViewportMask::all() : ViewportMask{};

    auto readColors = [&root]( const char* fieldName, Color& res )
    {
        const auto& colors = root["Colors"]["Faces"][fieldName];
        Vector4f resVec;
        deserializeFromJson( colors["Diffuse"], resVec );
        res = Color( resVec );
    };

    readColors( "SelectedMode", selectedColor_.get() );
    readColors( "UnselectedMode", unselectedColor_.get() );
    readColors( "BackFaces", backFacesColor_.get() );

    if ( root["Colors"]["GlobalAlpha"].isUInt() )
        globalAlpha_.get() = uint8_t( root["Colors"]["GlobalAlpha"].asUInt() );

    if ( const auto& showNameJson = root["ShowName"]; showNameJson.isUInt() )
        showName_ = ViewportMask( showNameJson.asUInt() );

    if ( root["UseDefaultSceneProperties"].isBool() && root["UseDefaultSceneProperties"].asBool() )
        setDefaultSceneProperties_();

    dirty_ = DIRTY_ALL;
}

Box3f VisualObject::getWorldBox( ViewportId id ) const
{
    return transformed( getBoundingBox(), worldXf( id ) );
}

size_t VisualObject::heapBytes() const
{
    return Object::heapBytes()
        + MR::heapBytes( renderObj_ );
}

std::vector<std::string> VisualObject::getInfoLines() const
{
    auto res = Object::getInfoLines();
    if ( renderObj_ )
        res.push_back( "GL mem: " + bytesString( renderObj_->glBytes() ) );
    return res;
}

void VisualObject::resetFrontColor()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectMesh ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectMesh ), false );
}

void VisualObject::resetColors()
{
    resetFrontColor();

    setBackColor( SceneColors::get( SceneColors::BackFaces ) );
}

void VisualObject::boundingBoxToInfoLines_( std::vector<std::string> & res ) const
{
    auto bbox = getBoundingBox();
    if ( bbox.valid() )
    {
        std::stringstream ss;
        ss << "box min: (" << bbox.min.x << ", " << bbox.min.y << ", " << bbox.min.z << ")";
        res.push_back( ss.str() );

        ss = std::stringstream{};
        ss << "box max: (" << bbox.max.x << ", " << bbox.max.y << ", " << bbox.max.z << ")";
        res.push_back( ss.str() );

        ss = std::stringstream{};
        const auto bcenter = bbox.center();
        ss << "box center: (" << bcenter.x << ", " << bcenter.y << ", " << bcenter.z << ")";
        res.push_back( ss.str() );

        ss = std::stringstream{};
        const auto bsize = bbox.size();
        ss << "(" << bsize.x << ", " << bsize.y << ", " << bsize.z << ")";
        const auto boxStr = ss.str();
        res.push_back( "box size: " + boxStr );

        const auto wbox = getWorldBox();
        if ( wbox.valid() )
        {
            const auto wbsize = wbox.size();
            ss = std::stringstream{};
            ss << "(" << wbsize.x << ", " << wbsize.y << ", " << wbsize.z << ")";
            const auto wboxStr = ss.str();
            if ( boxStr != wboxStr )
                res.push_back( "world box size: " + wboxStr );
        }
    }
    else
        res.push_back( "empty box" );
}

void VisualObject::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectMesh ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectMesh ), false );
    setBackColor( SceneColors::get( SceneColors::BackFaces ) );
}

void VisualObject::setDefaultSceneProperties_()
{
    setDefaultColors_();
}

} //namespace MR
