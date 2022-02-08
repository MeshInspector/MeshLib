#include "MRObjectLines.h"
#include "MRObjectFactory.h"
#include "MRPolyline.h"
#include "MRSerializer.h"
#include "MRTimer.h"
#include "MRMeshTexture.h"
#include "MRPch/MRJson.h"
#include "MRSceneColors.h"
#include "MRPch/MRTBB.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectLines )

void ObjectLines::applyScale( float scaleFactor )
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

std::shared_ptr<Object> ObjectLines::clone() const
{
    auto res = std::make_shared<ObjectLines>( ProtectedStruct{}, *this );
    if ( polyline_ )
        res->polyline_ = std::make_shared<Polyline>( *polyline_ );
    return res;
}

std::shared_ptr<Object> ObjectLines::shallowClone() const
{
    auto res = std::make_shared<ObjectLines>( ProtectedStruct{}, *this );
    if ( polyline_ )
        res->polyline_ = polyline_;
    return res;
}

void ObjectLines::setPolyline( const std::shared_ptr<Polyline>& polyline )
{
    polyline_ = polyline;
    setDirtyFlags( DIRTY_ALL );
}

void ObjectLines::setDirtyFlags( uint32_t mask )
{
    VisualObject::setDirtyFlags( mask );

    if ( ( mask & DIRTY_POSITION || mask & DIRTY_PRIMITIVES ) && polyline_ )
    {
        polyline_->invalidateCaches();
        totalLength_.reset();
    }
}

void ObjectLines::setLineWidth( float width )
{
    if ( width == lineWidth_ )
        return;
    lineWidth_ = width;
    needRedraw_ = true;
}

void ObjectLines::setPointSize( float size )
{
    if ( size == pointSize_ )
        return;
    pointSize_ = size;
    needRedraw_ = true;
}

void ObjectLines::swap( Object& other )
{
    if ( auto otherLines = other.asType<ObjectLines>() )
        std::swap( *this, *otherLines );
    else
        assert( false );
}

std::vector<std::string> ObjectLines::getInfoLines() const
{
    std::vector<std::string> res;
    res.push_back( "type : ObjectLines" );

    std::stringstream ss;
    if ( polyline_ )
    {
        ss << "vertices : " << polyline_->topology.numValidVerts();
        res.push_back( ss.str() );

        if ( !totalLength_ )
            totalLength_ = polyline_->totalLength();
        res.push_back( "total length : " + std::to_string( *totalLength_ ) );

        boundingBoxToInfoLines_( res );
    }
    else
    {
        res.push_back( "no polyline" );
    }
    
    return res;
}

AllVisualizeProperties ObjectLines::getAllVisualizeProperties() const
{
    AllVisualizeProperties res;
    res.resize( LinesVisualizePropertyType::LinesVisualizePropsCount );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = getVisualizePropertyMask( unsigned( i ) );
    return res;
}

const ViewportMask& ObjectLines::getVisualizePropertyMask( unsigned type ) const
{
    switch ( LinesVisualizePropertyType::Type( type ) )
    {
    case MR::LinesVisualizePropertyType::Points:
        return showPoints_;
    case MR::LinesVisualizePropertyType::Smooth:
        return smoothConnections_;
    default:
        return VisualObject::getVisualizePropertyMask( type );
    }
}

ObjectLines::ObjectLines()
{
    setDefaultColors_();
}

void ObjectLines::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_(root);

    root["ShowPoints"] = showPoints_.value();
    root["SmoothConnections"] = smoothConnections_.value();

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
    for ( UndirectedEdgeId ue{0}; ue < polyline_->topology.undirectedEdgeSize(); ++ue )
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
    root["Type"].append( ObjectLines::TypeName() ); // will be appended in derived calls
}

void ObjectLines::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_(root);

    if ( root["ShowPoints"].isUInt() )
        showPoints_ = ViewportMask{ root["ShowPoints"].asUInt() };
    if ( root["SmoothConnections"].isUInt() )
        smoothConnections_ = ViewportMask{ root["SmoothConnections"].asUInt() };

    const auto& polylineRoot = root["Polyline"];
    if ( !polylineRoot.isObject() )
        return;

    const auto& pointsRoot = polylineRoot["Points"];
    const auto& linesRoot = polylineRoot["Lines"];

    if ( !pointsRoot.isArray() || !linesRoot.isArray() )
        return;

    Polyline polyline;
    polyline.points.resize( pointsRoot.size() );
    for ( int i = 0; i < polyline.points.size(); ++i )
        deserializeFromJson( pointsRoot[i], polyline.points.vec_[i] );

    int maxVertId = -1;
    for ( int i = 0; i < (int)linesRoot.size(); ++i )
        maxVertId = std::max( maxVertId, linesRoot[i].asInt() );

    polyline.topology.vertResize( maxVertId + 1 );
    for ( int i = 0; i < (int)linesRoot.size(); i += 2 )
        polyline.topology.makeEdge( VertId( linesRoot[i].asInt() ), VertId( linesRoot[i + 1].asInt() ) );

    polyline_ = std::make_shared<Polyline>( std::move( polyline ) );
    setDirtyFlags( DIRTY_ALL );
}

Box3f ObjectLines::computeBoundingBox_() const
{
    if ( !polyline_ )
        return {};
    Box3f box;
    for ( const auto& p : polyline_->points )
        box.include( p );
    return box;
}

Box3f ObjectLines::computeBoundingBoxXf_() const
{
    if( !polyline_ )
        return {};
    Box3f box;
    auto wXf = worldXf();
    for( const auto& p : polyline_->points )
        box.include( wXf (p) );
    return box;
}

void ObjectLines::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<ObjectLines>( *this );
}

void ObjectLines::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectLines ) );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectLines ), false );
}

}
