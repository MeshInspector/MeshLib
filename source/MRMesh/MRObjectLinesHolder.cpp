#include "MRObjectLinesHolder.h"
#include "MRObjectFactory.h"
#include "MRSerializer.h"
#include "MRHeapBytes.h"
#include "MRPch/MRJson.h"
#include "MRSceneColors.h"
#include "MRPch/MRTBB.h"
#include <filesystem>
#include "MRPolylineComponents.h"

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
    }

    if ( mask & DIRTY_POSITION || mask & DIRTY_PRIMITIVES )
    {
        totalLength_.reset();
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
        + MR::heapBytes( polyline_ );
}

size_t ObjectLinesHolder::numComponents() const
{
    if ( !numComponents_ )
        numComponents_ = PolylineComponents::getNumComponents( polyline_->topology );

    return *numComponents_;
}

AllVisualizeProperties ObjectLinesHolder::getAllVisualizeProperties() const
{
    AllVisualizeProperties res;
    res.resize( LinesVisualizePropertyType::LinesVisualizePropsCount );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = getVisualizePropertyMask( unsigned( i ) );
    return res;
}

const ViewportMask& ObjectLinesHolder::getVisualizePropertyMask( unsigned type ) const
{
    switch ( type )
    {
    case LinesVisualizePropertyType::Points:
        return showPoints_;
    case LinesVisualizePropertyType::Smooth:
        return smoothConnections_;
    default:
        return VisualObject::getVisualizePropertyMask( type );
    }
}

ObjectLinesHolder::ObjectLinesHolder()
{
    setDefaultColors_();
}

void ObjectLinesHolder::serializeBaseFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    root["ShowPoints"] = showPoints_.value();
    root["SmoothConnections"] = smoothConnections_.value();

    root["ColoringType"] = ( coloringType_ == ColoringType::LinesColorMap ) ? "PerLine" : "Solid";
    serializeToJson( linesColorMap_.vec_, root["LineColors"] );
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
        if ( stype == "PerLine" )
            setColoringType( ColoringType::LinesColorMap );
    }
    deserializeFromJson( root["LineColors"], linesColorMap_.vec_ );
}

void ObjectLinesHolder::deserializeFields_( const Json::Value& root )
{
    deserializeBaseFields_( root );

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

}
