#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRObjectVoxels.h"
#include "MRObjectFactory.h"
#include "MRMesh.h"
#include "MRVDBConversions.h"
#include "MRFloatGrid.h"
#include "MRSimpleVolume.h"
#include "MRVoxelsSave.h"
#include "MRVoxelsLoad.h"
#include "MRSerializer.h"
#include "MRMeshNormals.h"
#include "MRTimer.h"
#include "MRPch/MRJson.h"
#include "MRSceneColors.h"
#include "MRStringConvert.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRAsyncLaunchType.h"
#include "MROpenVDBHelper.h"
#include <filesystem>
#include <thread>

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectVoxels )

constexpr size_t cVoxelsHistogramBinsNumber = 256;

void ObjectVoxels::construct( const SimpleVolume& volume, ProgressCallback cb )
{
    mesh_.reset();
    vdbVolume_.data = simpleVolumeToDenseGrid( volume, cb );
    vdbVolume_.dims = volume.dims;
    vdbVolume_.voxelSize = volume.voxelSize;
    indexer_ = VolumeIndexer( vdbVolume_.dims );
    activeBox_ = Box3i( Vector3i(), vdbVolume_.dims );
    reverseVoxelSize_ = { 1 / vdbVolume_.voxelSize.x,1 / vdbVolume_.voxelSize.y,1 / vdbVolume_.voxelSize.z };
    updateHistogram_( volume.min, volume.max );
    if ( volumeRendering_ )
        dirty_ |= ( DIRTY_PRIMITIVES | DIRTY_TEXTURE );
}

void ObjectVoxels::construct( const FloatGrid& grid, const Vector3f& voxelSize, ProgressCallback cb )
{
    if ( !grid )
        return;
    vdbVolume_.data = grid;

    auto vdbDims = vdbVolume_.data->evalActiveVoxelDim();
    vdbVolume_.dims = {vdbDims.x(),vdbDims.y(),vdbDims.z()};
    indexer_ = VolumeIndexer( vdbVolume_.dims );
    activeBox_ = Box3i( Vector3i(), vdbVolume_.dims );
    vdbVolume_.voxelSize = voxelSize;
    reverseVoxelSize_ = { 1 / vdbVolume_.voxelSize.x,1 / vdbVolume_.voxelSize.y,1 / vdbVolume_.voxelSize.z };

    updateHistogramAndSurface( cb );
    if ( volumeRendering_ )
        dirty_ |= ( DIRTY_PRIMITIVES | DIRTY_TEXTURE );
}

void ObjectVoxels::construct( const VdbVolume& volume, ProgressCallback cb )
{
    construct( volume.data, volume.voxelSize, cb );
}

void ObjectVoxels::updateHistogramAndSurface( ProgressCallback cb )
{
    if ( !vdbVolume_.data )
        return;

    float min{0.0f}, max{0.0f};

    evalGridMinMax( vdbVolume_.data, min, max );

    const float progressTo = ( mesh_ && cb ) ? 0.5f : 1.f;
    updateHistogram_( min, max, subprogress( cb, 0.f, progressTo ) );
    vdbVolume_.min = min;
    vdbVolume_.max = max;
    if ( mesh_ )
    {
        mesh_.reset();

        const float progressFrom = cb ? 0.5f : 0.f;
        setIsoValue( isoValue_, subprogress( cb, progressFrom, 1.f ) );
    }
}

tl::expected<bool, std::string> ObjectVoxels::setIsoValue( float iso, ProgressCallback cb, bool updateSurface )
{
    if ( !vdbVolume_.data )
        return false; // no volume presented in this
    if ( mesh_ && iso == isoValue_ )
        return false; // current iso surface represents required iso value

    isoValue_ = iso;
    if ( updateSurface )
    {
        auto recRes = recalculateIsoSurface( isoValue_, cb );
        if ( !recRes.has_value() )
            return tl::make_unexpected( recRes.error() );
        updateIsoSurface( *recRes );
    }
    if ( volumeRendering_ )
        dirty_ |= DIRTY_TEXTURE;
    return updateSurface;
}

std::shared_ptr<Mesh> ObjectVoxels::updateIsoSurface( std::shared_ptr<Mesh> mesh )
{
    if ( mesh != mesh_ )
    {
        mesh_.swap( mesh );
        setDirtyFlags( DIRTY_ALL );
        isoSurfaceChangedSignal();
    }
    return mesh;

}

VdbVolume ObjectVoxels::updateVdbVolume( VdbVolume vdbVolume )
{
    auto oldVdbVolume = std::move( vdbVolume_ );
    vdbVolume_ = std::move( vdbVolume );
    setDirtyFlags( DIRTY_ALL );
    return oldVdbVolume;
}

Histogram ObjectVoxels::updateHistogram( Histogram histogram )
{
    auto oldHistogram = std::move( histogram_ );
    histogram_ = std::move( histogram );
    return oldHistogram;
}

tl::expected<std::shared_ptr<Mesh>, std::string> ObjectVoxels::recalculateIsoSurface( float iso, ProgressCallback cb /*= {} */ )
{
    if ( !vdbVolume_.data )
        return tl::make_unexpected("No VdbVolume available");
    auto meshRes = gridToMesh( vdbVolume_.data, vdbVolume_.voxelSize, maxSurfaceTriangles_, iso, 0.0f, cb );
    if ( !meshRes.has_value() && meshRes.error() == "Operation was canceled." )
        return tl::make_unexpected( meshRes.error() );

    FloatGrid downsampledGrid = vdbVolume_.data;
    while ( !meshRes.has_value() )
    {
        downsampledGrid = resampled( downsampledGrid, 2.0f );
        meshRes = gridToMesh( std::move( downsampledGrid ), 2.0f * vdbVolume_.voxelSize, maxSurfaceTriangles_, iso, 0.0f, cb );
        if ( !meshRes.has_value() )
            return tl::make_unexpected( meshRes.error() );
    }
    return std::make_shared<Mesh>( std::move( meshRes.value() ) );
}

void ObjectVoxels::setActiveBounds( const Box3i& activeBox, ProgressCallback cb, bool updateSurface )
{
    if ( !vdbVolume_.data )
        return;
    if ( !activeBox.valid() )
        return;

    activeBox_ = activeBox;
    auto accessor = vdbVolume_.data->getAccessor();

    size_t counter = 0;
    float volume = float( vdbVolume_.dims.x ) * vdbVolume_.dims.y * vdbVolume_.dims.z;
    float cbModifier = updateSurface ? 0.5f : 1.0f;

    bool insideX = false;
    bool insideY = false;
    bool insideZ = false;
    for ( int z = 0; z < vdbVolume_.dims.z; ++z )
    for ( int y = 0; y < vdbVolume_.dims.y; ++y )
    for ( int x = 0; x < vdbVolume_.dims.x; ++x )
    {
        insideX = ( x >= activeBox_.min.x && x < activeBox_.max.x );
        insideY = ( y >= activeBox_.min.y && y < activeBox_.max.y );
        insideZ = ( z >= activeBox_.min.z && z < activeBox_.max.z );
        accessor.setActiveState( {x,y,z}, insideX && insideY && insideZ );
        reportProgress( cb, [&]{ return cbModifier * float( counter ) / volume; }, ++counter, 256 );
    }
    if ( updateSurface )
    {
        ProgressCallback isoProgressCallback = subprogress( cb, cbModifier, 1.0f );
        auto recRes = recalculateIsoSurface( isoValue_, isoProgressCallback );
        std::shared_ptr<Mesh> recMesh;
        if ( recRes.has_value() )
            recMesh = *recRes;
        updateIsoSurface( recMesh );
    }
}

VoxelId ObjectVoxels::getVoxelIdByCoordinate( const Vector3i& coord ) const
{
    return indexer_.toVoxelId( coord );
}

VoxelId ObjectVoxels::getVoxelIdByPoint( const Vector3f& point ) const
{
    return getVoxelIdByCoordinate( Vector3i( mult( point, reverseVoxelSize_ ) ) );
}

Vector3i ObjectVoxels::getCoordinateByVoxelId( VoxelId id ) const
{
    return indexer_.toPos( id );
}

bool ObjectVoxels::prepareDataForVolumeRendering( ProgressCallback cb /*= {} */ ) const
{
    if ( !vdbVolume_.data )
        return false;
    volumeRenderingData_ = std::make_unique<SimpleVolumeU8>();
    auto& res = *volumeRenderingData_;
    res.max = vdbVolume_.max;
    res.min = vdbVolume_.min;
    res.voxelSize = vdbVolume_.voxelSize;
    auto activeBox = vdbVolume_.data->evalActiveVoxelBoundingBox();
    res.dims = Vector3i( activeBox.dim().x(), activeBox.dim().y(), activeBox.dim().z() );
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    const auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> cancelled{ false };
    std::atomic<size_t> finishedVoxels{ 0 };

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, indexer.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        auto accessor = vdbVolume_.data->getConstAccessor();
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            if ( cb && cancelled.load( std::memory_order_relaxed ) )
                return;
            auto coord = indexer.toPos( VoxelId( i ) );
            auto vdbCoord = openvdb::Coord( coord.x + activeBox.min().x(), coord.y + activeBox.min().y(), coord.z + activeBox.min().z() );
            res.data[i] = uint8_t( std::clamp( ( accessor.getValue( vdbCoord ) - res.min ) / ( res.max - res.min ), 0.0f, 1.0f ) * 255.0f );
        }
        if ( cb )
        {
            finishedVoxels.fetch_add( range.size(), std::memory_order_relaxed );
            if ( std::this_thread::get_id() == mainThreadId &&
                !cb( float( finishedVoxels.load( std::memory_order_relaxed ) / float( indexer.size() ) ) ) )
                cancelled.store( true, std::memory_order_relaxed );
        }
    } );
    if ( !cancelled )
        return true;
    volumeRenderingData_.reset();
    return false;
}

void ObjectVoxels::enableVolumeRendering( bool on )
{
    if ( volumeRendering_ == on )
        return;
    volumeRendering_ = on;
    if ( volumeRendering_ )
    {
        if ( !volumeRenderingData_ )
            prepareDataForVolumeRendering();
        renderObj_ = createRenderObject<ObjectVoxels>( *this );
    }
    else
    {
        renderObj_ = createRenderObject<ObjectMeshHolder>( *this );
    }
    setDirtyFlags( DIRTY_ALL );
}

bool ObjectVoxels::hasVisualRepresentation() const
{
    if ( isVolumeRenderingEnabled() )
        return false;
    return ObjectMeshHolder::hasVisualRepresentation();
}

void ObjectVoxels::setMaxSurfaceTriangles( int maxFaces )
{
    if ( maxFaces == maxSurfaceTriangles_ )
        return;
    maxSurfaceTriangles_ = maxFaces;
    if ( !mesh_ || mesh_->topology.numValidFaces() <= maxSurfaceTriangles_ )
        return;
    mesh_.reset();
    setIsoValue( isoValue_ );
}

std::shared_ptr<Object> ObjectVoxels::clone() const
{
    auto res = std::make_shared<ObjectVoxels>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    if ( vdbVolume_.data )
        res->vdbVolume_.data = MakeFloatGrid( vdbVolume_.data->deepCopy() );
    return res;
}

std::shared_ptr<Object> ObjectVoxels::shallowClone() const
{
    auto res = std::make_shared<ObjectVoxels>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    if ( vdbVolume_.data )
        res->vdbVolume_ = vdbVolume_;
    return res;
}

void ObjectVoxels::setDirtyFlags( uint32_t mask )
{
    VisualObject::setDirtyFlags( mask );

    if ( ( mask & DIRTY_POSITION || mask & DIRTY_FACE ) && mesh_ )
        mesh_->invalidateCaches();
}

size_t ObjectVoxels::heapBytes() const
{
    return ObjectMeshHolder::heapBytes()
        + vdbVolume_.heapBytes()
        + histogram_.heapBytes()
        + MR::heapBytes( volumeRenderingData_ );
}

void ObjectVoxels::swapBase_( Object& other )
{
    if ( auto otherVoxels = other.asType<ObjectVoxels>() )
        std::swap( *this, *otherVoxels );
    else
        assert( false );
}

void ObjectVoxels::swapSignals_( Object& other )
{
    ObjectMeshHolder::swapSignals_( other );
    if ( auto otherVoxels = other.asType<ObjectVoxels>() )
        std::swap( isoSurfaceChangedSignal, otherVoxels->isoSurfaceChangedSignal );
    else
        assert( false );
}

/// @brief class to parallel reduce histogram calculation
template<typename TreeT>
class HistogramCalcProc
{
public:
    using ValueT = typename TreeT::ValueType;
    using TreeAccessor = openvdb::tree::ValueAccessor<const TreeT>;
    using LeafIterT = typename TreeT::LeafCIter;
    using TileIterT = typename TreeT::ValueAllCIter;

    HistogramCalcProc( float min, float max ) :
        hist( min, max, cVoxelsHistogramBinsNumber )
    {}

    HistogramCalcProc( const HistogramCalcProc& other ) :
        hist( Histogram( other.hist.getMin(), other.hist.getMax(), cVoxelsHistogramBinsNumber ) )
    {}

    void action( const LeafIterT&, const TreeAccessor& treeAcc, const openvdb::math::CoordBBox& bbox )
    {
        for ( auto it = bbox.begin(); it != bbox.end(); ++it )
        {
            ValueT value = ValueT();
            if ( treeAcc.probeValue( *it, value ) )
                hist.addSample( value );
        }
    }

    void action( const TileIterT& iter, const TreeAccessor&, const openvdb::math::CoordBBox& bbox )
    {
        ValueT value = iter.getValue();
        const size_t count = size_t( bbox.volume() );
        hist.addSample( value, count );
    }

    void join( const HistogramCalcProc& other )
    {
        hist.addHistogram( other.hist );
    }

    Histogram hist;
};


void ObjectVoxels::updateHistogram_( float min, float max, ProgressCallback cb /*= {}*/ )
{
    MR_TIMER;

    RangeSize size = calculateRangeSize( *vdbVolume_.data );
    
    using HistogramCalcProcFT = HistogramCalcProc<openvdb::FloatTree>;
    HistogramCalcProcFT histCalcProc( min, max );
    using HistRangeProcessorOne = RangeProcessorSingle<openvdb::FloatTree, HistogramCalcProcFT>;
    HistRangeProcessorOne calc( vdbVolume_.data->evalActiveVoxelBoundingBox(), vdbVolume_.data->tree(), histCalcProc );
    

    std::function<bool( size_t, size_t )> progressFn;
    if ( size.tile )
    {
        typename HistRangeProcessorOne::TileIterT tileIterMain = vdbVolume_.data->tree().cbeginValueAll();
        tileIterMain.setMaxDepth( tileIterMain.getLeafDepth() - 1 ); // skip leaf nodes
        typename HistRangeProcessorOne::TileRange tileRangeMain( tileIterMain );

        std::atomic<size_t> tileDone = 0;
        if ( cb )
        {
            if ( !size.leaf )
                progressFn = [cb, tileSize = size.tile, &tileDone]( size_t, size_t t )
                {
                    tileDone += t;
                    return !cb( float( tileDone ) / tileSize );
                };
            else
                progressFn = [cb, tileSize = size.tile, &tileDone]( size_t, size_t t )
                {
                    tileDone += t;
                    return !cb( float( tileDone ) / tileSize / 2.f );
                };
            calc.setProgressFn( progressFn );
        }
        tbb::parallel_reduce( tileRangeMain, calc );
    }

    if ( size.leaf )
    {
        typename HistRangeProcessorOne::LeafRange leafRangeMain( vdbVolume_.data->tree().cbeginLeaf() );
        std::atomic<size_t> leafDone = 0;
        if ( cb )
        {
            if ( !size.tile )
                progressFn = [cb, leafSize = size.leaf, &leafDone]( size_t l, size_t )
                {
                    leafDone += l;
                    return !cb( float( leafDone ) / leafSize );
                };
            else
                progressFn = [cb, leafSize = size.leaf, &leafDone]( size_t l, size_t )
                {
                    leafDone += l;
                    return !cb( float( leafDone ) / leafSize / 2.f + 0.5f );
                };
            calc.setProgressFn( progressFn );
        }
        tbb::parallel_reduce( leafRangeMain, calc );
    }

    histogram_ = std::move( calc.mProc.hist );
}

void ObjectVoxels::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectVoxels ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectVoxels ), false );
}

ObjectVoxels::ObjectVoxels( const ObjectVoxels& other ) :
    ObjectMeshHolder( other )
{
    vdbVolume_.dims = other.vdbVolume_.dims;
    isoValue_ = other.isoValue_;
    histogram_ = other.histogram_;
    vdbVolume_.voxelSize = other.vdbVolume_.voxelSize;
    activeBox_ = other.activeBox_;

    indexer_ = other.indexer_;
    reverseVoxelSize_ = other.reverseVoxelSize_;
}

ObjectVoxels::ObjectVoxels()
{
    setDefaultColors_();
}

void ObjectVoxels::applyScale( float scaleFactor )
{
    vdbVolume_.voxelSize *= scaleFactor;

    ObjectMeshHolder::applyScale( scaleFactor );
}

void ObjectVoxels::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );
    serializeToJson( vdbVolume_.voxelSize, root["VoxelSize"] );

    serializeToJson( vdbVolume_.dims, root["Dimensions"] );
    serializeToJson( activeBox_.min, root["MinCorner"] );
    serializeToJson( activeBox_.max, root["MaxCorner"] );
    serializeToJson( selectedVoxels_, root["SelectionVoxels"] );

    root["IsoValue"] = isoValue_;
    root["Type"].append( ObjectVoxels::TypeName() );
}

tl::expected<std::future<void>, std::string> ObjectVoxels::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !vdbVolume_.data )
        return {};

    return std::async( getAsyncLaunchType(),
        [this, filename = utf8string( path ) + ".raw"]() { MR::VoxelsSave::saveRaw( filename, vdbVolume_ ); } );
}

void ObjectVoxels::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );
    if ( root["VoxelSize"].isDouble() )
        vdbVolume_.voxelSize = Vector3f::diagonal( ( float )root["VoxelSize"].asDouble() );
    else
        deserializeFromJson( root["VoxelSize"], vdbVolume_.voxelSize );

    deserializeFromJson( root["Dimensions"], vdbVolume_.dims );

    deserializeFromJson( root["MinCorner"], activeBox_.min );

    deserializeFromJson( root["MaxCorner"], activeBox_.max );

    deserializeFromJson( root["SelectionVoxels"], selectedVoxels_ );

    if ( root["IsoValue"].isNumeric() )
        isoValue_ = root["IsoValue"].asFloat();

    if ( !activeBox_.valid() )
        activeBox_ = Box3i( Vector3i(), vdbVolume_.dims );

    if ( activeBox_.min != Vector3i() || activeBox_.max != vdbVolume_.dims )
        setActiveBounds( activeBox_ );
    else 
        setIsoValue( isoValue_ );
}

#ifndef MRMESH_NO_DICOM
VoidOrErrStr ObjectVoxels::deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb )
{
    auto res = VoxelsLoad::loadRaw( utf8string( path ) + ".raw", progressCb );
    if ( !res.has_value() )
        return tl::make_unexpected( res.error() );
    
    construct( res.value().data, res.value().voxelSize );
    if ( !vdbVolume_.data )
        return tl::make_unexpected( "No grid loaded" );

    return {};
}
#endif

std::vector<std::string> ObjectVoxels::getInfoLines() const
{
    std::vector<std::string> res = ObjectMeshHolder::getInfoLines();
    res.push_back( "dims: (" + std::to_string( vdbVolume_.dims.x ) + ", " + std::to_string( vdbVolume_.dims.y ) + ", " + std::to_string( vdbVolume_.dims.z ) + ")" );
    res.push_back( "voxel size: (" + std::to_string( vdbVolume_.voxelSize.x ) + ", " + std::to_string( vdbVolume_.voxelSize.y ) + ", " + std::to_string( vdbVolume_.voxelSize.z ) + ")" );
    res.push_back( "iso-value: " + std::to_string( isoValue_ ) );
    return res;
}

}
#endif
