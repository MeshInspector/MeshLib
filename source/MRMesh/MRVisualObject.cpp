#include "MRVisualObject.h"
#include "MRObjectFactory.h"
#include "MRSerializer.h"
#include "MRSceneColors.h"
#include "MRMesh.h"
#include "MRObjectMesh.h"
#include "MRTimer.h"
#include "MRHeapBytes.h"
#include "MRStringConvert.h"
#include <filesystem>
#include <tl/expected.hpp>
#include "MRPch/MRJson.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( VisualObject )

VisualObject::VisualObject()
{
    setDefaultColors_();
}

VisualObject::VisualObject( const VisualObject& other ):
    Object( other )
{
    showTexture_ = other.showTexture_;
    clipByPlane_ = other.clipByPlane_;
    showLabels_ = other.showLabels_;
    showName_ = other.showName_;
    cropLabels_ = other.cropLabels_;
    pickable_ = other.pickable_;
    invertNormals_ = other.invertNormals_;

    labelsColor_ = other.labelsColor_;

    shininess_ = other.shininess_;

    coloringType_ = other.coloringType_;
    vertsColorMap_ = other.vertsColorMap_;
    selectedColor_ = other.selectedColor_;
    unselectedColor_ = other.unselectedColor_;
    backFacesColor_ = other.backFacesColor_;
    depthTest_ = other.depthTest_;
    texture_ = other.texture_;
    uvCoordinates_ = other.uvCoordinates_;
    labels_ = other.labels_;
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
        getVisualizePropertyMask_( unsigned( i ) ) = properties[i];
}

AllVisualizeProperties VisualObject::getAllVisualizeProperties() const
{
    AllVisualizeProperties res;
    res.resize( VisualizeMaskType::VisualizePropsCount );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = getVisualizePropertyMask( unsigned( i ) );
    return res;
}

const Color& VisualObject::getFrontColor( bool selected /*= true */ ) const
{
    return selected ? selectedColor_ : unselectedColor_;
}

void VisualObject::setFrontColor( const Color& color, bool selected /*= true */ )
{
    auto& oldColor = selected ? selectedColor_ : unselectedColor_;
    if ( oldColor == color )
        return;

    oldColor = color;
}

const Color& VisualObject::getBackColor() const
{
    return backFacesColor_;
}

void VisualObject::setBackColor( const Color& color )
{
    if ( backFacesColor_ == color )
        return;
    backFacesColor_ = color;
    dirty_ |= DIRTY_BACK_FACES;
}

const Color& VisualObject::getLabelsColor() const
{
    return labelsColor_;
}

void VisualObject::setLabelsColor( const Color& color )
{
    labelsColor_ = color;
    needRedraw_ = true;
}

void VisualObject::setDirtyFlags( uint32_t mask )
{
    if ( mask & DIRTY_POSITION )
        mask |= DIRTY_RENDER_NORMALS | DIRTY_BOUNDING_BOX | DIRTY_BORDER_LINES | DIRTY_EDGES_SELECTION;
    if ( mask & DIRTY_FACE )
        mask |= DIRTY_RENDER_NORMALS | DIRTY_BORDER_LINES | DIRTY_EDGES_SELECTION | DIRTY_POSITION;
    // DIRTY_POSITION because we use corner rendering and need to update render verts

    dirty_ |= mask;
}

const uint32_t& VisualObject::getDirtyFlags() const
{
    return dirty_;
}

void VisualObject::resetDirty() const
{
    // Bounding box and normals (all caches) is cleared only if it was recounted
    dirty_ &= DIRTY_CACHES;
}

Box3f VisualObject::getBoundingBox() const
{
    std::unique_lock lock( readCacheMutex_.getMutex() );
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
    switch ( VisualizeMaskType::Type( type ) )
    {
    case MR::VisualizeMaskType::Visibility:
        return visibilityMask_;
    case MR::VisualizeMaskType::Texture:
        return showTexture_;
    case MR::VisualizeMaskType::InvertedNormals:
        return invertNormals_;
    case MR::VisualizeMaskType::Labels:
        return showLabels_;
    case MR::VisualizeMaskType::ClippedByPlane:
        return clipByPlane_;
    case MR::VisualizeMaskType::Name:
        return showName_;
    case MR::VisualizeMaskType::CropLabelsByViewportRect:
        return cropLabels_;
    case MR::VisualizeMaskType::DepthTest:
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
    root["ShowLabes"] = showLabels();

    auto writeColors = [&root]( const char * fieldName, const Color& val )
    {
        auto& colors = root["Colors"]["Faces"][fieldName];
        serializeToJson( Vector4f( val ), colors["Diffuse"] );// To support old version 
    };

    writeColors( "SelectedMode", selectedColor_ );
    writeColors( "UnselectedMode", unselectedColor_ );
    writeColors( "BackFaces", backFacesColor_ );

    // labels
    serializeToJson( Vector4f( labelsColor_ ), root["Colors"]["Labels"] );

    // append base type
    root["Type"].append( VisualObject::TypeName() );
}

void VisualObject::deserializeFields_( const Json::Value& root )
{
    Object::deserializeFields_( root );

    if ( root["InvertNormals"].isBool() ) // Support old versions
        invertNormals_ = root["InvertNormals"].asBool() ? ViewportMask::all() : ViewportMask{};
    if ( root["ShowLabes"].isBool() )
        showLabels( root["ShowLabes"].asBool() );

    auto readColors = [&root]( const char* fieldName, Color& res )
    {
        const auto& colors = root["Colors"]["Faces"][fieldName];
        Vector4f resVec;
        deserializeFromJson( colors["Diffuse"], resVec );
        res = Color( resVec );
    };

    readColors( "SelectedMode", selectedColor_ );
    readColors( "UnselectedMode", unselectedColor_ );
    readColors( "BackFaces", backFacesColor_ );

    Vector4f resVec;
    // labels
    deserializeFromJson( root["Colors"]["Labels"], resVec );
    labelsColor_ = Color( resVec );

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
        + texture_.heapBytes()
        + uvCoordinates_.heapBytes()
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
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectMesh ) );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectMesh ), false );
    setBackColor( SceneColors::get( SceneColors::BackFaces ) );
    setLabelsColor( SceneColors::get( SceneColors::Labels ) );
}

} //namespace MR
