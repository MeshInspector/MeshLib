#include "MRObjectLinesHolder.h"
#include "MRPolyline.h"
#include "MRObjectFactory.h"
#include "MRSerializer.h"
#include "MRHeapBytes.h"
#include "MRSceneColors.h"
#include "MRPolylineComponents.h"
#include "MRParallelFor.h"
#include "MRDirectory.h"
#include "MRLinesLoad.h"
#include "MRPch/MRJson.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectLinesHolder )

void ObjectLinesHolder::applyScale( float scaleFactor )
{
    if ( !polyline_ )
        return;

    auto& points = polyline_->points;

    tbb::parallel_for( tbb::blocked_range<int>( 0, ( int )points.size() ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            points[VertId( i )] *= scaleFactor;
        }
    } );
    setDirtyFlags( DIRTY_POSITION );
}

bool ObjectLinesHolder::hasVisualRepresentation() const
{
    return polyline_ && polyline_->topology.numValidVerts() != 0;
}

std::shared_ptr<Object> ObjectLinesHolder::clone() const
{
    auto res = std::make_shared<ObjectLinesHolder>( ProtectedStruct{}, *this );
    if ( polyline_ )
        res->polyline_ = std::make_shared<Polyline3>( *polyline_ );
    return res;
}

std::shared_ptr<Object> ObjectLinesHolder::shallowClone() const
{
    auto res = std::make_shared<ObjectLinesHolder>( ProtectedStruct{}, *this );
    if ( polyline_ )
        res->polyline_ = polyline_;
    return res;
}

void ObjectLinesHolder::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    VisualObject::setDirtyFlags( mask, invalidateCaches );

    if ( mask & DIRTY_PRIMITIVES )
    {
        numComponents_.reset();
        numUndirectedEdges_.reset();
    }

    if ( mask & DIRTY_POSITION || mask & DIRTY_PRIMITIVES )
    {
        totalLength_.reset();
        avgEdgeLen_.reset();
        worldBox_.reset();
        worldBox_.get().reset();
        if ( invalidateCaches && polyline_ )
            polyline_->invalidateCaches();
    }
}

void ObjectLinesHolder::setLineWidth( float width )
{
    if ( width == lineWidth_ )
        return;
    lineWidth_ = width;
    needRedraw_ = true;
}

void ObjectLinesHolder::setPointSize( float size )
{
    if ( size == pointSize_ )
        return;
    pointSize_ = size;
    needRedraw_ = true;
}

void ObjectLinesHolder::swapBase_( Object& other )
{
    if ( auto otherLines = other.asType<ObjectLinesHolder>() )
        std::swap( *this, *otherLines );
    else
        assert( false );
}

Box3f ObjectLinesHolder::getWorldBox( ViewportId id ) const
{
    if ( !polyline_ )
        return {};
    bool isDef = true;
    const auto worldXf = this->worldXf( id, &isDef );
    if ( isDef )
        id = {};
    auto & cache = worldBox_[id];
    if ( auto v = cache.get( worldXf ) )
        return *v;
    const auto box = polyline_->computeBoundingBox( &worldXf );
    cache.set( worldXf, box );
    return box;
}

size_t ObjectLinesHolder::heapBytes() const
{
    return VisualObject::heapBytes()
        + linesColorMap_.heapBytes()
        + vertsColorMap_.heapBytes()
        + MR::heapBytes( polyline_ );
}

float ObjectLinesHolder::avgEdgeLen() const
{
    if ( !avgEdgeLen_ )
        avgEdgeLen_ = polyline_ ? polyline_->averageEdgeLength() : 0;

    return *avgEdgeLen_;
}

size_t ObjectLinesHolder::numUndirectedEdges() const
{
    if ( !numUndirectedEdges_ )
        numUndirectedEdges_ = polyline_ ? polyline_->topology.computeNotLoneUndirectedEdges() : 0;
    return *numUndirectedEdges_;
}

size_t ObjectLinesHolder::numComponents() const
{
    if ( !numComponents_ )
        numComponents_ = PolylineComponents::getNumComponents( polyline_->topology );

    return *numComponents_;
}

float ObjectLinesHolder::totalLength() const
{
    if ( !totalLength_ )
        totalLength_ = polyline_ ? polyline_->totalLength() : 0.f;
    return *totalLength_;
}

void ObjectLinesHolder::resetFrontColor()
{
    // cannot implement in the opposite way to keep `setDefaultColors_()` non-virtual
    setDefaultColors_();
}

bool ObjectLinesHolder::supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const
{
    return VisualObject::supportsVisualizeProperty( type ) || type.tryGet<LinesVisualizePropertyType>().has_value();
}

AllVisualizeProperties ObjectLinesHolder::getAllVisualizeProperties() const
{
    AllVisualizeProperties ret = VisualObject::getAllVisualizeProperties();
    getAllVisualizePropertiesForEnum<LinesVisualizePropertyType>( ret );
    return ret;
}

void ObjectLinesHolder::setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos )
{
    VisualObject::setAllVisualizeProperties_( properties, pos );
    setAllVisualizePropertiesForEnum<LinesVisualizePropertyType>( properties, pos );
}

const ViewportMask& ObjectLinesHolder::getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const
{
    if ( auto value = type.tryGet<LinesVisualizePropertyType>() )
    {
        switch ( *value )
        {
        case LinesVisualizePropertyType::Points:
            return showPoints_;
        case LinesVisualizePropertyType::Smooth:
            return smoothConnections_;
        case LinesVisualizePropertyType::_count: break; // MSVC warns if this is missing, despite `[[maybe_unused]]` on the `_count`.
        }
        assert( false && "Invalid enum." );
        return visibilityMask_;
    }
    else
    {
        return VisualObject::getVisualizePropertyMask( type );
    }
}

void ObjectLinesHolder::copyColors( const ObjectLinesHolder& src, const VertMap& thisToSrc )
{
    MR_TIMER;

    setColoringType( src.getColoringType() );

    const auto& srcColorMap = src.getVertsColorMap();
    if ( srcColorMap.empty() )
        return;

    VertColors colorMap;
    colorMap.resizeNoInit( thisToSrc.size() );
    ParallelFor( colorMap, [&] ( VertId id )
    {
        auto curId = thisToSrc[id];
        if( curId.valid() )
            colorMap[id] = srcColorMap[curId];
    } );
    setVertsColorMap( std::move( colorMap ) );
}

ObjectLinesHolder::ObjectLinesHolder()
{
    setDefaultSceneProperties_();
}

void ObjectLinesHolder::serializeBaseFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    root["ShowPoints"] = showPoints_.value();
    root["SmoothConnections"] = smoothConnections_.value();

    switch( coloringType_ )
    {
    case ColoringType::VertsColorMap:
        root["ColoringType"] = "PerVertex";
        break;
    case ColoringType::LinesColorMap:
        root["ColoringType"] = "PerLine";
        break;
    default:
        root["ColoringType"] = "Solid";
    }

    serializeToJson( linesColorMap_.vec_, root["LineColors"] );
    serializeToJson( vertsColorMap_.vec_, root["VertColors"] );

    root["LineWidth"] = lineWidth_;
}

void ObjectLinesHolder::serializeFields_( Json::Value& root ) const
{
    serializeBaseFields_( root );

    if ( !polyline_ )
        return;
    auto& polylineRoot = root["Polyline"];
    auto& pointsRoot = polylineRoot["Points"];
    auto& linesRoot = polylineRoot["Lines"];
    for ( const auto& p : polyline_->points )
    {
        Json::Value val;
        serializeToJson( p, val );
        pointsRoot.append( val );
    }
    for ( UndirectedEdgeId ue{ 0 }; ue < polyline_->topology.undirectedEdgeSize(); ++ue )
    {
        auto o = polyline_->topology.org( ue );
        auto d = polyline_->topology.dest( ue );
        if ( o && d )
        {
            linesRoot.append( int( o ) );
            linesRoot.append( int( d ) );
        }
    }

    // Type
    root["Type"].append( ObjectLinesHolder::TypeName() ); // will be appended in derived calls
}

Expected<void> ObjectLinesHolder::deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb )
{
    // currently we do not write polyline in a separate model file;
    // the code below is for the future, when we start doing it;
    // for now it simply returns without error but with null polyline_
    polyline_.reset();
    vertsColorMap_.clear();

    auto modelPath = findPathWithExtension( path );
    if ( modelPath.empty() )
        return {};

    auto res = LinesLoad::fromAnySupportedFormat( modelPath, { .colors = &vertsColorMap_, .callback = progressCb } );
    if ( !res.has_value() )
        return unexpected( res.error() );

    polyline_ = std::make_shared<Polyline3>( std::move( res.value() ) );
    return {};
}

void ObjectLinesHolder::deserializeBaseFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );

    if ( root["ShowPoints"].isUInt() )
        showPoints_ = ViewportMask{ root["ShowPoints"].asUInt() };
    if ( root["SmoothConnections"].isUInt() )
        smoothConnections_ = ViewportMask{ root["SmoothConnections"].asUInt() };

    if ( root["ColoringType"].isString() )
    {
        const auto stype = root["ColoringType"].asString();
        if ( stype == "PerVertex" )
            setColoringType( ColoringType::VertsColorMap );
        else if ( stype == "PerLine" )
            setColoringType( ColoringType::LinesColorMap );
    }
    deserializeFromJson( root["LineColors"], linesColorMap_.vec_ );
    deserializeFromJson( root["VertColors"], vertsColorMap_.vec_ );

    if ( root["UseDefaultSceneProperties"].isBool() && root["UseDefaultSceneProperties"].asBool() )
        setDefaultSceneProperties_();

    if ( const auto& lineWidthJson = root["LineWidth"]; lineWidthJson.isDouble() )
        lineWidth_ = float( lineWidthJson.asDouble() );
}

void ObjectLinesHolder::deserializeFields_( const Json::Value& root )
{
    deserializeBaseFields_( root );

    if ( polyline_ )
    {
        // polyline already loaded in deserializeModel_
        setDirtyFlags( DIRTY_ALL );
        return;
    }

    const auto& polylineRoot = root["Polyline"];
    if ( !polylineRoot.isObject() )
        return;

    const auto& pointsRoot = polylineRoot["Points"];
    const auto& linesRoot = polylineRoot["Lines"];

    if ( !pointsRoot.isArray() || !linesRoot.isArray() )
        return;

    Polyline3 polyline;
    polyline.points.resize( pointsRoot.size() );
    for ( int i = 0; i < polyline.points.size(); ++i )
        deserializeFromJson( pointsRoot[i], polyline.points.vec_[i] );

    int maxVertId = -1;
    for ( int i = 0; i < ( int )linesRoot.size(); ++i )
        maxVertId = std::max( maxVertId, linesRoot[i].asInt() );

    polyline.topology.vertResize( maxVertId + 1 );
    for ( int i = 0; i < ( int )linesRoot.size(); i += 2 )
        polyline.topology.makeEdge( VertId( linesRoot[i].asInt() ), VertId( linesRoot[i + 1].asInt() ) );

    polyline_ = std::make_shared<Polyline3>( std::move( polyline ) );
    setDirtyFlags( DIRTY_ALL );
}

Box3f ObjectLinesHolder::computeBoundingBox_() const
{
    if ( !polyline_ )
        return Box3f();
    return polyline_->computeBoundingBox();
}

void ObjectLinesHolder::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<ObjectLinesHolder>( *this );
}

void ObjectLinesHolder::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectLines ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectLines ), false );
}

void ObjectLinesHolder::setDefaultSceneProperties_()
{
    setDefaultColors_();
}

}
