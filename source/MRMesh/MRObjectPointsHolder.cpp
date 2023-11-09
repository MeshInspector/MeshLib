#include "MRObjectPointsHolder.h"
#include "MRObjectFactory.h"
#include "MRBitSetParallelFor.h"
#include "MRPointsSave.h"
#include "MRPointsLoad.h"
#include "MRSceneColors.h"
#include "MRHeapBytes.h"
#include "MRSerializer.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRAsyncLaunchType.h"
#include "MRStringConvert.h"
#include <filesystem>

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectPointsHolder )

ObjectPointsHolder::ObjectPointsHolder()
{
    setDefaultColors_();
}

void ObjectPointsHolder::applyScale( float scaleFactor )
{
    if ( !points_ )
        return;

    auto& points = points_->points;

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

bool ObjectPointsHolder::hasVisualRepresentation() const
{
    return points_ && points_->validPoints.any();
}

std::shared_ptr<MR::Object> ObjectPointsHolder::clone() const
{
    auto res = std::make_shared<ObjectPointsHolder>( ProtectedStruct{}, *this );
    if ( points_ )
        res->points_ = std::make_shared<PointCloud>( *points_ );
    return res;
}

std::shared_ptr<MR::Object> ObjectPointsHolder::shallowClone() const
{
    auto res = std::make_shared<ObjectPointsHolder>( ProtectedStruct{}, *this );
    if ( points_ )
        res->points_ = points_;
    return res;
}

void ObjectPointsHolder::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    VisualObject::setDirtyFlags( mask, invalidateCaches );

    if ( mask & DIRTY_FACE )
        numValidPoints_.reset();

    if ( mask & DIRTY_POSITION || mask & DIRTY_FACE )
    {
        worldBox_.reset();
        worldBox_.get().reset();
        if ( invalidateCaches && points_ )
            points_->invalidateCaches();
    }
}

void ObjectPointsHolder::selectPoints( VertBitSet newSelection )
{
    selectedPoints_ = std::move( newSelection );
    numSelectedPoints_.reset();
    dirty_ |= DIRTY_SELECTION;
}

void ObjectPointsHolder::setSelectedVerticesColor( const Color& color, ViewportId id )
{
    if ( color == selectedVerticesColor_.get( id ) )
        return;
    selectedVerticesColor_.set( color, id );
    needRedraw_ = true;
}

AllVisualizeProperties ObjectPointsHolder::getAllVisualizeProperties() const
{
    AllVisualizeProperties res;
    res.resize( PointsVisualizePropertyType::PointsVisualizePropsCount );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = getVisualizePropertyMask( unsigned( i ) );
    return res;
}

const ViewportMask& ObjectPointsHolder::getVisualizePropertyMask( unsigned type ) const
{
    switch ( type )
    {
    case PointsVisualizePropertyType::SelectedVertices:
        return showSelectedVertices_;
    default:
        return VisualObject::getVisualizePropertyMask( type );
    }
}

void ObjectPointsHolder::setPointSize( float size )
{
    if ( pointSize_ == size )
        return;

    pointSize_ = size;
    needRedraw_ = true;
}

Box3f ObjectPointsHolder::getWorldBox( ViewportId id ) const
{
    if ( !points_ )
        return {};
    bool isDef = true;
    const auto worldXf = this->worldXf( id, &isDef );
    if ( isDef )
        id = {};
    auto & cache = worldBox_[id];
    if ( auto v = cache.get( worldXf ) )
        return *v;
    const auto box = points_->computeBoundingBox( &worldXf );
    cache.set( worldXf, box );
    return box;
}

size_t ObjectPointsHolder::numValidPoints() const
{
    if ( !numValidPoints_ )
        numValidPoints_ = points_ ? points_->validPoints.count() : 0;

    return *numValidPoints_;
}

size_t ObjectPointsHolder::numSelectedPoints() const
{
    if ( !numSelectedPoints_ )
        numSelectedPoints_ = selectedPoints_.count();

    return *numSelectedPoints_;
}

size_t ObjectPointsHolder::heapBytes() const
{
    return VisualObject::heapBytes()
        + selectedPoints_.heapBytes()
        + MR::heapBytes( points_ );
}

void ObjectPointsHolder::swapBase_( Object& other )
{
    if ( auto otherPointsHolder = other.asType<ObjectPointsHolder>() )
        std::swap( *this, *otherPointsHolder );
    else
        assert( false );
}

Box3f ObjectPointsHolder::computeBoundingBox_() const
{
    if ( !points_ )
        return {};
    tbb::enumerable_thread_specific<Box3f> threadData;
    BitSetParallelFor( points_->validPoints, [&] ( VertId id )
    {
        threadData.local().include( points_->points[id] );
    } );
    Box3f bb;
    for ( const auto& b : threadData )
        bb.include( b );
    return bb;
}

Expected<std::future<void>, std::string> ObjectPointsHolder::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !points_ )
        return {};

    const auto * colorMapPtr = vertsColorMap_.empty() ? nullptr : &vertsColorMap_;
#ifndef MRMESH_NO_OPENCTM
    return std::async( getAsyncLaunchType(),
        [points = points_, filename = utf8string( path ) + ".ctm", ptr = colorMapPtr] ()
    {
        MR::PointsSave::toCtm( *points, pathFromUtf8( filename ), { MR::PointsSave::Settings{ .saveValidOnly = false, .colors = ptr } } );
    } );
#else
    return std::async( getAsyncLaunchType(),
        [points = points_, filename = utf8string( path ) + ".ply", ptr = colorMapPtr] ()
    {
        MR::PointsSave::toPly( *points, pathFromUtf8( filename ), MR::PointsSave::Settings{ .saveValidOnly = false, .colors = ptr } );
    } );
#endif
}

VoidOrErrStr ObjectPointsHolder::deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb )
{
#ifndef MRMESH_NO_OPENCTM
    auto res = PointsLoad::fromCtm( pathFromUtf8( utf8string( path ) + ".ctm" ), &vertsColorMap_, progressCb );
#else
    auto res = PointsLoad::fromPly( pathFromUtf8( utf8string( path ) + ".ply" ), &vertsColorMap_, progressCb );
#endif
    if ( !res.has_value() )
        return unexpected( res.error() );

    if ( !vertsColorMap_.empty() )
        setColoringType( ColoringType::VertsColorMap );

    points_ = std::make_shared<PointCloud>( std::move( res.value() ) );
    return {};
}

void ObjectPointsHolder::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    serializeToJson( Vector4f( selectedVerticesColor_.get() ), root["Colors"]["Selection"]["Points"] );
    serializeToJson( selectedPoints_, root["SelectionVertBitSet"] );
    if ( points_ )
        serializeToJson( points_->validPoints, root["ValidVertBitSet"] );
}

void ObjectPointsHolder::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );

    Vector4f resVec;
    deserializeFromJson( root["Colors"]["Selection"]["Points"], resVec );
    selectedVerticesColor_.set( Color( resVec ) );

    deserializeFromJson( root["SelectionVertBitSet"], selectedPoints_ );
    if ( points_ )
        deserializeFromJson( root["ValidVertBitSet"], points_->validPoints );
}

void ObjectPointsHolder::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<ObjectPointsHolder>( *this );
}

void ObjectPointsHolder::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectPoints ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectPoints ), false );
    setSelectedVerticesColor( SceneColors::get( SceneColors::SelectedPoints ) );
}

const ViewportProperty<Color>& ObjectPointsHolder::getSelectedVerticesColorsForAllViewports() const
{
    return selectedColor_;
}

void ObjectPointsHolder::setSelectedVerticesColorsForAllViewports( ViewportProperty<Color> val )
{
    selectedColor_ = std::move( val );
}

}
