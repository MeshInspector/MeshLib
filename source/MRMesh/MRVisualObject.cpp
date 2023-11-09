#include "MRVisualObject.h"
#include "MRObjectFactory.h"
#include "MRSerializer.h"
#include "MRSceneColors.h"
#include "MRMesh.h"
#include "MRObjectMesh.h"
#include "MRTimer.h"
#include "MRHeapBytes.h"
#include "MRStringConvert.h"
#include "MRExpected.h"
#include "MRParallelFor.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRSuppressWarning.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( VisualObject )

VisualObject::VisualObject()
{
    setDefaultColors_();
}

void VisualObject::setVisualizeProperty( bool value, unsigned type, ViewportMask viewportMask )
{
    auto res = getVisualizePropertyMask( type );
    if ( value )
        res |= viewportMask;
    else
        res &= ~viewportMask;

    setVisualizePropertyMask( type, res );
}

void VisualObject::setVisualizePropertyMask( unsigned type, ViewportMask viewportMask )
{
    auto& mask = getVisualizePropertyMask_( type );
    if ( mask == viewportMask )
        return;
    mask = viewportMask;
    needRedraw_ = true;
}

bool VisualObject::getVisualizeProperty( unsigned type, ViewportMask viewportMask ) const
{
    return !( getVisualizePropertyMask( type ) & viewportMask ).empty();
}

void VisualObject::toggleVisualizeProperty( unsigned type, ViewportMask viewportMask )
{
    setVisualizePropertyMask( type, getVisualizePropertyMask( type ) ^ viewportMask );
}

void VisualObject::setAllVisualizeProperties( const AllVisualizeProperties& properties )
{
    for ( int i = 0; i < properties.size(); ++i )
        setVisualizePropertyMask( unsigned( i ), properties[i] );
}

AllVisualizeProperties VisualObject::getAllVisualizeProperties() const
{
    AllVisualizeProperties res;
    res.resize( VisualizeMaskType::VisualizePropsCount );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = getVisualizePropertyMask( unsigned( i ) );
    return res;
}

const Color& VisualObject::getFrontColor( bool selected /*= true */, ViewportId viewportId /*= {} */ ) const
{
    return selected ? selectedColor_.get( viewportId ) : unselectedColor_.get( viewportId );
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
    return backFacesColor_.get( viewportId );
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
    return globalAlpha_.get( viewportId );
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

const Color& VisualObject::getLabelsColor( ViewportId viewportId ) const
{
    return labelsColor_.get( viewportId );
}

void VisualObject::setLabelsColor( const Color& color, ViewportId viewportId )
{
    labelsColor_.set( color, viewportId );
    needRedraw_ = true;
}

const ViewportProperty<Color>& VisualObject::getLabelsColorsForAllViewports() const
{
    return labelsColor_;
}

void VisualObject::setLabelsColorsForAllViewports( ViewportProperty<Color> val )
{
    labelsColor_ = val;
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
}

void VisualObject::resetDirty() const
{
    // Bounding box and normals (all caches) is cleared only if it was recounted
    dirty_ &= DIRTY_CACHES;
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

void VisualObject::copyColors( const VisualObject & src, const VertMap & thisToSrc )
{
    MR_TIMER

    setColoringType( src.getColoringType() );

    const auto& srcColorMap = src.getVertsColorMap();
    if ( srcColorMap.empty() )
        return;

    VertColors colorMap;
    colorMap.resizeNoInit( thisToSrc.size() );
    ParallelFor( colorMap, [&]( VertId id )
    {
        colorMap[id] = srcColorMap[thisToSrc[id]];
    } );
    setVertsColorMap( std::move( colorMap ) );
}

std::shared_ptr<Object> VisualObject::clone() const
{
    return std::make_shared<VisualObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> VisualObject::shallowClone() const
{
    return clone();
}

void VisualObject::render( const RenderParams& params ) const
{
    setupRenderObject_();
    if ( !renderObj_ )
        return;

    renderObj_->render( params );
}

void VisualObject::renderForPicker( const BaseRenderParams& params, unsigned id) const
{
    setupRenderObject_();
    if ( !renderObj_ )
        return;

    renderObj_->renderPicker( params, id );
}

void VisualObject::bindAllVisualization() const
{
    setupRenderObject_();
    if ( !renderObj_ )
        return;

    renderObj_->forceBindAll();
}

void VisualObject::swapBase_( Object& other )
{    
    if ( auto otherVis = other.asType<VisualObject>() )
        std::swap( *this, *otherVis );
    else
        assert( false );
}

ViewportMask& VisualObject::getVisualizePropertyMask_( unsigned type )
{
    return const_cast< ViewportMask& >( getVisualizePropertyMask( type ) );
}

const ViewportMask& VisualObject::getVisualizePropertyMask( unsigned type ) const
{
    switch ( type )
    {
    case VisualizeMaskType::Visibility:
        return visibilityMask_;
    case VisualizeMaskType::InvertedNormals:
        return invertNormals_;
    case VisualizeMaskType::Labels:
        return showLabels_;
    case VisualizeMaskType::ClippedByPlane:
        return clipByPlane_;
    case VisualizeMaskType::Name:
        return showName_;
    case VisualizeMaskType::CropLabelsByViewportRect:
        return cropLabels_;
    case VisualizeMaskType::DepthTest:
        return depthTest_;
    default:
        assert( false );
        return visibilityMask_;
    }
}

void VisualObject::serializeFields_( Json::Value& root ) const
{
    Object::serializeFields_( root );
    root["InvertNormals"] = !invertNormals_.empty();
MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
    root["ShowLabes"] = showLabels();
MR_SUPPRESS_WARNING_POP

    auto writeColors = [&root]( const char * fieldName, const Color& val )
    {
        auto& colors = root["Colors"]["Faces"][fieldName];
        serializeToJson( Vector4f( val ), colors["Diffuse"] );// To support old version 
    };

    writeColors( "SelectedMode", selectedColor_.get() );
    writeColors( "UnselectedMode", unselectedColor_.get() );
    writeColors( "BackFaces", backFacesColor_.get() );

    root["Colors"]["GlobalAlpha"] = globalAlpha_.get();

    // labels
    serializeToJson( Vector4f( labelsColor_.get() ), root["Colors"]["Labels"] );

    // append base type
    root["Type"].append( VisualObject::TypeName() );
}

void VisualObject::deserializeFields_( const Json::Value& root )
{
    Object::deserializeFields_( root );

    if ( root["InvertNormals"].isBool() ) // Support old versions
        invertNormals_ = root["InvertNormals"].asBool() ? ViewportMask::all() : ViewportMask{};
MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
    if ( root["ShowLabes"].isBool() )
        showLabels( root["ShowLabes"].asBool() );
MR_SUPPRESS_WARNING_POP
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

    Vector4f resVec;
    // labels
    deserializeFromJson( root["Colors"]["Labels"], resVec );
    labelsColor_.set( Color( resVec ) );

    dirty_ = DIRTY_ALL;
}

Box3f VisualObject::getWorldBox( ViewportId id ) const
{
    return transformed( getBoundingBox(), worldXf( id ) );
}

size_t VisualObject::heapBytes() const
{
    return Object::heapBytes()
        + vertsColorMap_.heapBytes()
        + MR::heapBytes( labels_ )
        + MR::heapBytes( renderObj_ );
}

std::vector<std::string> VisualObject::getInfoLines() const
{
    auto res = Object::getInfoLines();
    if ( renderObj_ )
        res.push_back( "GL mem: " + bytesString( renderObj_->glBytes() ) );
    return res;
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
MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
    setLabelsColor( SceneColors::get( SceneColors::Labels ) );
MR_SUPPRESS_WARNING_POP
}

} //namespace MR
