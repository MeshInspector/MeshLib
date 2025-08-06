#include "MRObjectPointsHolder.h"
#include "MRIOFormatsRegistry.h"
#include "MRObjectFactory.h"
#include "MRBitSetParallelFor.h"
#include "MRPointsSave.h"
#include "MRPointsLoad.h"
#include "MRSceneColors.h"
#include "MRHeapBytes.h"
#include "MRSerializer.h"
#include "MRStringConvert.h"
#include "MRDirectory.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRAsyncLaunchType.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectPointsHolder )

ObjectPointsHolder::ObjectPointsHolder()
{
    setDefaultSceneProperties_();
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
    {
        numValidPoints_.reset();
        updateRenderDiscretization_();
    }

    if ( mask & DIRTY_POSITION || mask & DIRTY_FACE )
    {
        worldBox_.reset();
        worldBox_.get().reset();
        if ( invalidateCaches && points_ )
            points_->invalidateCaches();
    }
}


void ObjectPointsHolder::swapSignals_( Object& other )
{
    VisualObject::swapSignals_( other );
    if ( auto otherPoints = other.asType<ObjectPointsHolder>() )
    {
        std::swap( pointsSelectionChangedSignal, otherPoints->pointsSelectionChangedSignal );
    }
    else
        assert( false );
}

void ObjectPointsHolder::updateSelectedPoints( VertBitSet& selection )
{
    std::swap( selectedPoints_, selection );
    numSelectedPoints_.reset();
    pointsSelectionChangedSignal();
    setDirtyFlags( DIRTY_SELECTION );
}

const VertBitSet& ObjectPointsHolder::getSelectedPointsOrAll() const
{
    return ( !points_ || numSelectedPoints() ) ? selectedPoints_ : points_->validPoints;
}

void ObjectPointsHolder::setSelectedVerticesColor( const Color& color, ViewportId id )
{
    if ( color == selectedVerticesColor_.get( id ) )
        return;
    selectedVerticesColor_.set( color, id );
    needRedraw_ = true;
}

bool ObjectPointsHolder::supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const
{
    return VisualObject::supportsVisualizeProperty( type ) || type.tryGet<PointsVisualizePropertyType>().has_value();
}

void ObjectPointsHolder::copyColors( const ObjectPointsHolder & src, const VertMap & thisToSrc, const FaceMap& )
{
    MR_TIMER;

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

AllVisualizeProperties ObjectPointsHolder::getAllVisualizeProperties() const
{
    AllVisualizeProperties ret = VisualObject::getAllVisualizeProperties();
    getAllVisualizePropertiesForEnum<PointsVisualizePropertyType>( ret );
    return ret;
}

void ObjectPointsHolder::setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos )
{
    VisualObject::setAllVisualizeProperties_( properties, pos );
    setAllVisualizePropertiesForEnum<PointsVisualizePropertyType>( properties, pos );
}

const ViewportMask &ObjectPointsHolder::getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const
{
    if ( auto value = type.tryGet<PointsVisualizePropertyType>() )
    {
        switch ( *value )
        {
        case PointsVisualizePropertyType::SelectedVertices:
            return showSelectedVertices_;
        case PointsVisualizePropertyType::_count: break; // MSVC warns if this is missing, despite `[[maybe_unused]]` on the `_count`.
        }
        assert( false && "Invalid enum." );
        return visibilityMask_;
    }
    else
    {
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

size_t ObjectPointsHolder::numRenderingValidPoints() const
{
    if ( !points_ )
        return 0;

    return ( points_->validPoints.find_last() + 1 ) / renderDiscretization_;
}

size_t ObjectPointsHolder::heapBytes() const
{
    return VisualObject::heapBytes()
        + selectedPoints_.heapBytes()
        + vertsColorMap_.heapBytes()
        + MR::heapBytes( points_ );
}

void ObjectPointsHolder::setMaxRenderingPoints( int val )
{
    if ( maxRenderingPoints_ == val )
        return;
    maxRenderingPoints_ = val;
    updateRenderDiscretization_();
}

void ObjectPointsHolder::setSerializeFormat( const char * newFormat )
{
    if ( newFormat && *newFormat != '.' )
    {
        assert( false );
        return;
    }
    serializeFormat_ = newFormat;
}

void ObjectPointsHolder::resetFrontColor()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectPoints ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectPoints ), false );
}

void ObjectPointsHolder::resetColors()
{
    // cannot implement in the opposite way to keep `setDefaultColors_()` non-virtual
    setDefaultColors_();
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

Expected<std::future<Expected<void>>> ObjectPointsHolder::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !points_ )
        return {};

    if ( points_->points.empty() ) // some formats (e.g. .ctm) require at least one point in the vector
        return std::async( getAsyncLaunchType(), []{ return Expected<void>{}; } );

    SaveSettings saveSettings;
    saveSettings.onlyValidPoints = false;
    saveSettings.packPrimitives = false;
    if ( !vertsColorMap_.empty() )
        saveSettings.colors = &vertsColorMap_;
    auto save = [points = points_, serializeFormat = serializeFormat_ ? serializeFormat_ : defaultSerializePointsFormat(), path, saveSettings]()
    {
        auto filename = path;
        const auto extension = std::string( "*" ) + serializeFormat;
        if ( auto pointsSaver = PointsSave::getPointsSaver( extension ); pointsSaver.fileSave != nullptr )
        {
            filename += serializeFormat;
            return pointsSaver.fileSave( *points, filename, saveSettings );
        }
        else
        {
            filename += ".ply";
            return MR::PointsSave::toAnySupportedFormat( *points, filename, saveSettings );
        }
    };
    return std::async( getAsyncLaunchType(), save );
}

Expected<void> ObjectPointsHolder::deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb )
{
    auto modelPath = pathFromUtf8( utf8string( path ) + ".ctm" ); //quick path for most used format
    std::error_code ec;
    if ( !is_regular_file( modelPath, ec ) )
        modelPath = findPathWithExtension( path );
    if ( modelPath.empty()                   // now we do not write a file for empty point cloud
        || file_size( modelPath, ec ) == 0 ) // and previously an empty file was created
    {
        points_ = std::make_shared<PointCloud>();
        return {};
    }
    auto res = PointsLoad::fromAnySupportedFormat( modelPath, {
        .colors = &vertsColorMap_,
        .callback = progressCb,
    } );
    if ( !res.has_value() )
        return unexpected( std::move( res.error() ) );

    if ( !vertsColorMap_.empty() )
        setColoringType( ColoringType::VertsColorMap );

    points_ = std::make_shared<PointCloud>( std::move( res.value() ) );
    //updateRenderDiscretization_(); must be called later after valid points are deserialized
    return {};
}

void ObjectPointsHolder::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );

    serializeToJson( Vector4f( selectedVerticesColor_.get() ), root["Colors"]["Selection"]["Points"] );
    serializeToJson( selectedPoints_, root["SelectionVertBitSet"] );
    if ( points_ )
        serializeToJson( points_->validPoints, root["ValidVertBitSet"] );

    root["PointSize"] = pointSize_;
    root["MaxRenderingPoints"] = maxRenderingPoints_;
}

void ObjectPointsHolder::deserializeFields_( const Json::Value& root )
{
    MR_TIMER;
    VisualObject::deserializeFields_( root );

    Vector4f resVec;
    deserializeFromJson( root["Colors"]["Selection"]["Points"], resVec );
    selectedVerticesColor_.set( Color( resVec ) );

    deserializeFromJson( root["SelectionVertBitSet"], selectedPoints_ );
    if ( points_ )
    {
        deserializeFromJson( root["ValidVertBitSet"], points_->validPoints );
        numValidPoints_.reset();
    }
    updateRenderDiscretization_();

    if ( root["UseDefaultSceneProperties"].isBool() && root["UseDefaultSceneProperties"].asBool() )
        setDefaultSceneProperties_();

    if ( const auto& pointSizeJson = root["PointSize"]; pointSizeJson.isDouble() )
        pointSize_ = float( pointSizeJson.asDouble() );

    if ( root["MaxRenderingPoints"].isInt() )
    {
        maxRenderingPoints_ = root["MaxRenderingPoints"].asInt();
        updateRenderDiscretization_();
    }
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

void ObjectPointsHolder::setDefaultSceneProperties_()
{
    setDefaultColors_();
}

void ObjectPointsHolder::updateRenderDiscretization_()
{
    int newRenderDiscretization = maxRenderingPoints_ <= 0 ? 1 :
        int( numValidPoints() + maxRenderingPoints_ - 1 ) / maxRenderingPoints_;
    newRenderDiscretization = std::max( 1, newRenderDiscretization );
    if ( newRenderDiscretization == renderDiscretization_ )
        return;
    renderDiscretization_ = newRenderDiscretization;
    needRedraw_ = true;
    renderDiscretizationChangedSignal();
}

// .PLY format is the most compact among other formats with zero compression costs
static std::string sDefaultSerializePointsFormat = ".ply";

const std::string & defaultSerializePointsFormat()
{
    return sDefaultSerializePointsFormat;
}

void setDefaultSerializePointsFormat( std::string newFormat )
{
    assert( !newFormat.empty() && newFormat[0] == '.' );
    sDefaultSerializePointsFormat = std::move( newFormat );
}

} //namespace MR
