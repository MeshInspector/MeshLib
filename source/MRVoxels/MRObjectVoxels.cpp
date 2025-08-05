#include "MRObjectVoxels.h"
#include "MRVDBConversions.h"
#include "MRVDBFloatGrid.h"
#include "MRFloatGrid.h"
#include "MRVoxelsSave.h"
#include "MRVoxelsLoad.h"
#include "MROpenVDBHelper.h"

#include "MRMesh/MRObjectFactory.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRMeshNormals.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRParallelMinMax.h"
#include "MRMesh/MRDirectory.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRAsyncLaunchType.h"
#include "MRPch/MRFmt.h"
#include <filesystem>
#include <thread>

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectVoxels )

constexpr size_t cVoxelsHistogramBinsNumber = 256;

void ObjectVoxels::construct( const SimpleVolume& simpleVolume, const std::optional<Vector2f> & minmax, ProgressCallback cb, bool normalPlusGrad )
{
    data_.mesh.reset();
    activeVoxels_.reset();
    activeBounds_.reset();
    if ( minmax )
    {
        vdbVolume_.min = minmax->x;
        vdbVolume_.max = minmax->y;
    }
    else
        std::tie( vdbVolume_.min, vdbVolume_.max ) = parallelMinMax( simpleVolume.data );
    vdbVolume_.data = simpleVolumeToDenseGrid( simpleVolume, normalPlusGrad ? vdbVolume_.max : vdbVolume_.min, cb );
    vdbVolume_.dims = simpleVolume.dims;
    vdbVolume_.voxelSize = simpleVolume.voxelSize;

    indexer_ = VolumeIndexer( vdbVolume_.dims );
    reverseVoxelSize_ = { 1 / vdbVolume_.voxelSize.x,1 / vdbVolume_.voxelSize.y,1 / vdbVolume_.voxelSize.z };

    if ( normalPlusGrad )
        vdbVolume_.data->setGridClass( openvdb::GRID_LEVEL_SET );

    volumeRenderActiveVoxels_.clear();

    updateHistogram_( vdbVolume_.min, vdbVolume_.max );
    uint32_t dirtyMask = DIRTY_VOLUME;
    if ( volumeRendering_ )
        dirtyMask |= DIRTY_PRIMITIVES | DIRTY_TEXTURE | DIRTY_SELECTION;
    setDirtyFlags( dirtyMask );
}

void ObjectVoxels::construct( const SimpleVolumeMinMax& simpleVolumeMinMax, ProgressCallback cb, bool normalPlusGrad )
{
    construct( simpleVolumeMinMax, Vector2f( simpleVolumeMinMax.min, simpleVolumeMinMax.max ), cb, normalPlusGrad );
}

void ObjectVoxels::construct( const FloatGrid& grid, const Vector3f& voxelSize, const std::optional<Vector2f> & minmax )
{
    assert( grid );
    if ( !grid )
        return;
    activeVoxels_.reset();
    activeBounds_.reset();
    vdbVolume_.data = grid;
    vdbVolume_.dims = fromVdb( vdbVolume_.data->evalActiveVoxelDim() );
    indexer_ = VolumeIndexer( vdbVolume_.dims );
    vdbVolume_.voxelSize = voxelSize;
    if ( minmax )
    {
        vdbVolume_.min = minmax->x;
        vdbVolume_.max = minmax->y;
    }
    else
        evalGridMinMax( vdbVolume_.data, vdbVolume_.min, vdbVolume_.max );
    reverseVoxelSize_ = { 1 / vdbVolume_.voxelSize.x,1 / vdbVolume_.voxelSize.y,1 / vdbVolume_.voxelSize.z };

    volumeRenderActiveVoxels_.clear();

    updateHistogram_( vdbVolume_.min, vdbVolume_.max );
    uint32_t dirtyMask = DIRTY_VOLUME;
    if ( volumeRendering_ )
        dirtyMask |= DIRTY_PRIMITIVES | DIRTY_TEXTURE | DIRTY_SELECTION;
    setDirtyFlags( dirtyMask );
}

void ObjectVoxels::construct( const VdbVolume& volume )
{
    construct( volume.data, volume.voxelSize, Vector2f( volume.min, volume.max ) );
}

void ObjectVoxels::updateHistogramAndSurface( ProgressCallback cb )
{
    if ( !vdbVolume_.data )
        return;

    float min{0.0f}, max{0.0f};

    evalGridMinMax( vdbVolume_.data, min, max );

    const float progressTo = ( data_.mesh && cb ) ? 0.5f : 1.f;
    updateHistogram_( min, max, subprogress( cb, 0.f, progressTo ) );
    vdbVolume_.min = min;
    vdbVolume_.max = max;
    if ( data_.mesh )
    {
        data_.mesh.reset();

        const float progressFrom = cb ? 0.5f : 0.f;
        (void)setIsoValue( isoValue_, subprogress( cb, progressFrom, 1.f ) ); //TODO: propagate error outside
    }
}

Expected<bool> ObjectVoxels::setIsoValue( float iso, ProgressCallback cb, bool updateSurface )
{
    if ( !vdbVolume_.data )
        return false; // no volume presented in this
    if ( data_.mesh && iso == isoValue_ )
        return false; // current iso surface represents required iso value

    isoValue_ = iso;
    if ( updateSurface )
    {
        auto recRes = recalculateIsoSurface( isoValue_, cb );
        if ( !recRes.has_value() )
            return unexpected( recRes.error() );
        updateIsoSurface( *recRes );
    }
    if ( volumeRendering_ )
        setDirtyFlags( DIRTY_TEXTURE );
    return updateSurface;
}

std::shared_ptr<Mesh> ObjectVoxels::updateIsoSurface( std::shared_ptr<Mesh> mesh )
{
    if ( mesh != data_.mesh )
    {
        data_.mesh.swap( mesh );
        setDirtyFlags( DIRTY_ALL );
        isoSurfaceChangedSignal();
    }
    return mesh;

}

VdbVolume ObjectVoxels::updateVdbVolume( VdbVolume vdbVolume )
{
    auto oldVdbVolume = std::move( vdbVolume_ );
    activeVoxels_.reset();
    activeBounds_.reset();
    vdbVolume_ = std::move( vdbVolume );
    indexer_ = VolumeIndexer( vdbVolume_.dims );
    reverseVoxelSize_ = { 1 / vdbVolume_.voxelSize.x, 1 / vdbVolume_.voxelSize.y, 1 / vdbVolume_.voxelSize.z };
    volumeRenderActiveVoxels_.clear();
    setDirtyFlags( DIRTY_ALL );
    return oldVdbVolume;
}

Histogram ObjectVoxels::updateHistogram( Histogram histogram )
{
    auto oldHistogram = std::move( histogram_ );
    histogram_ = std::move( histogram );
    return oldHistogram;
}

Expected<std::shared_ptr<Mesh>> ObjectVoxels::recalculateIsoSurface( float iso, MR::ProgressCallback cb ) const
{
    return recalculateIsoSurface( vdbVolume_, iso, cb );
}

Expected<std::shared_ptr<Mesh>> ObjectVoxels::recalculateIsoSurface( const VdbVolume& vdbVolumeCopy, float iso, ProgressCallback cb /*= {} */ ) const
{
    MR_TIMER;
    auto vdbVolume = vdbVolumeCopy;
    if ( !vdbVolume.data )
        return unexpected("No VdbVolume available");

    float startProgress = 0;   // where the current iteration has started
    float reachedProgress = 0; // maximum progress reached so far
    ProgressCallback myCb;
    if ( cb )
        myCb = [&startProgress, &reachedProgress, cb]( float p )
        {
            reachedProgress = startProgress + ( 1 - startProgress ) * p;
            return cb( reachedProgress );
        };

    for (;;)
    {
        // continue progress bar from the value where it stopped on the previous iteration
        startProgress = reachedProgress;
        Expected<Mesh> meshRes;
        if ( dualMarchingCubes_ )
        {
            meshRes = gridToMesh( vdbVolume.data, GridToMeshSettings{
                .voxelSize = vdbVolume.voxelSize,
                .isoValue = iso,
                .maxVertices = maxSurfaceVertices_,
                .cb = myCb
            } );
        }
        else
        {
            MarchingCubesParams vparams;
            vparams.iso = iso;
            vparams.maxVertices = maxSurfaceVertices_;
            vparams.cb = myCb;
            vparams.positioner = positioner_;
            if ( vdbVolume.data->getGridClass() == openvdb::GridClass::GRID_LEVEL_SET )
                vparams.lessInside = true;
            meshRes = marchingCubes( vdbVolume, vparams );
        }
        if ( meshRes.has_value() )
            return std::make_shared<Mesh>( std::move( meshRes.value() ) );
        if ( !meshRes.has_value() && meshRes.error() == stringOperationCanceled() )
            return unexpectedOperationCanceled();
        vdbVolume.data = resampled( vdbVolume.data, 2.0f );
        vdbVolume.voxelSize *= 2.0f;
        vdbVolume.dims = fromVdb( vdbVolume.data->evalActiveVoxelDim() );
    }
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


Histogram ObjectVoxels::recalculateHistogram( std::optional<Vector2f> minmax, ProgressCallback cb ) const
{
    RangeSize size = calculateRangeSize( *vdbVolume_.data );

    float min, max;
    if ( minmax )
    {
        min = minmax->x;
        max = minmax->y;
    }
    else
    {
        evalGridMinMax( vdbVolume_.data, min, max );
    }

    using HistogramCalcProcFT = HistogramCalcProc<openvdb::FloatTree>;
    HistogramCalcProcFT histCalcProc( min, max );
    using HistRangeProcessorOne = RangeProcessorSingle<openvdb::FloatTree, HistogramCalcProcFT>;
    HistRangeProcessorOne calc( vdbVolume_.data->evalActiveVoxelBoundingBox(), vdbVolume_.data->tree(), histCalcProc );

    if ( size.tile > 0 )
    {
        typename HistRangeProcessorOne::TileIterT tileIterMain = vdbVolume_.data->tree().cbeginValueAll();
        tileIterMain.setMaxDepth( tileIterMain.getLeafDepth() - 1 ); // skip leaf nodes
        typename HistRangeProcessorOne::TileRange tileRangeMain( tileIterMain );
        auto sb = size.leaf > 0 ? subprogress( cb, 0.0f, 0.5f ) : cb;
        calc.setProgressHolder( std::make_shared<RangeProgress>( sb, size.tile, RangeProgress::Mode::Tiles ) );
        tbb::parallel_reduce( tileRangeMain, calc );
    }

    if ( size.leaf > 0 )
    {
        typename HistRangeProcessorOne::LeafRange leafRangeMain( vdbVolume_.data->tree().cbeginLeaf() );
        auto sb = size.tile > 0 ? subprogress( cb, 0.5f, 1.0f ) : cb;
        calc.setProgressHolder( std::make_shared<RangeProgress>( sb, size.leaf, RangeProgress::Mode::Leaves ) );
        tbb::parallel_reduce( leafRangeMain, calc );
    }

    return calc.mProc.hist;
}

void ObjectVoxels::setDualMarchingCubes( bool on, bool updateSurface, ProgressCallback cb )
{
    MR_TIMER;
    dualMarchingCubes_ = on;
    if ( updateSurface )
    {
        auto recRes = recalculateIsoSurface( isoValue_, cb );
        if ( recRes.has_value() )
            updateIsoSurface( *recRes );
    }
}

void ObjectVoxels::setActiveBounds( const Box3i& activeBox, ProgressCallback cb, bool updateSurface )
{
    if ( !vdbVolume_.data )
        return;
    if ( !activeBox.valid() )
        return;

    float cbModifier = 1.0f;
    if ( updateSurface && volumeRendering_ )
        cbModifier = 1.0f / 3.0f;
    else if ( updateSurface || volumeRendering_ )
        cbModifier = 1.0f / 2.0f;
    float lastProgress = 0.0f;

    openvdb::CoordBBox activeVdbBox;
    activeVdbBox.min() = openvdb::Coord( activeBox.min.x, activeBox.min.y, activeBox.min.z );
    activeVdbBox.max() = openvdb::Coord( activeBox.max.x - 1, activeBox.max.y - 1, activeBox.max.z - 1 );

    // create active mask tree
    openvdb::TopologyTree topologyTree;

    reportProgress( cb, cbModifier * 0.25f );

    // update topology tree with new active box
    topologyTree.sparseFill( activeVdbBox, true );

    reportProgress( cb, cbModifier * 0.5f );

    // deactivate all of current grid
    openvdb::tools::foreach( vdbVolume_.data->tree().beginValueOn(), [] ( const openvdb::FloatTree::ValueOnIter& iter )
    {
        iter.setActiveState( false );
    }, false ); // looks like this operation is not safe to do in threaded mode

    reportProgress( cb, cbModifier * 0.75f );

    // copy valid topology to our tree part
    vdbVolume_.data->tree().topologyUnion( topologyTree );

    reportProgress( cb, cbModifier );

    // not safe to call from progress bar thread
    if ( !cb ) // we assume that cb presence indicates thread: if cb is set then it is progress bar thread, otherwise it is UI thread
        invalidateActiveBoundsCaches();

    lastProgress = cbModifier;
    if ( updateSurface )
    {
        ProgressCallback isoProgressCallback = subprogress( cb, lastProgress, 2.0f * cbModifier );
        lastProgress = 2.0f * cbModifier;
        auto recRes = recalculateIsoSurface( isoValue_, isoProgressCallback );
        std::shared_ptr<Mesh> recMesh;
        if ( recRes.has_value() )
            recMesh = *recRes;
        updateIsoSurface( recMesh );
    }
    uint32_t dirtyMask = DIRTY_VOLUME;
    if ( volumeRendering_ )
    {
        prepareDataForVolumeRendering( subprogress( cb, lastProgress, 1.0f ) );
        dirtyMask |= DIRTY_PRIMITIVES;
    }
    setDirtyFlags( dirtyMask );

}

void ObjectVoxels::invalidateActiveBoundsCaches()
{
    volumeRenderActiveVoxels_.clear();
    setDirtyFlags( DIRTY_SELECTION );
    activeVoxels_.reset();
    activeBounds_.reset();
}

const Box3i& ObjectVoxels::getActiveBounds() const
{
    if ( !activeBounds_ )
    {
        auto activeBox = vdbVolume_.data->evalActiveVoxelBoundingBox();
        auto min = fromVdb( activeBox.min() );
        auto max = fromVdb( activeBox.max() ) + Vector3i::diagonal( 1 );
        for ( int i = 0; i < 3; ++i )
        {
            // we should clamp values, because actual active box may lay outside of [0,dims), that we do not count in algorithms
            if ( min[i] < 0 ) min[i] = 0;
            if ( max[i] > vdbVolume_.dims[i] ) max[i] = vdbVolume_.dims[i];
        }
        activeBounds_.emplace( min, max );
    }
    return *activeBounds_;
}

void ObjectVoxels::setVolumeRenderActiveVoxels( const VoxelBitSet& activeVoxels )
{
    auto box = getActiveBounds().size();
    const bool valid = activeVoxels.empty() || activeVoxels.size() == box.x * box.y * box.z;
    assert( valid );
    if ( !valid )
        return;
    volumeRenderActiveVoxels_ = activeVoxels;
    setDirtyFlags( DIRTY_SELECTION );
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
    auto res = vdbVolumeToSimpleVolumeNorm( vdbVolume_, getActiveBounds(), {}, cb );
    if ( !res || res->data.empty() )
    {
        volumeRenderingData_.reset();
        return false;
    }
    volumeRenderingData_ = std::make_unique<SimpleVolume>( std::move( *res ) );
    return true;
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

void ObjectVoxels::setVolumeRenderingParams( const VolumeRenderingParams& params )
{
    if ( params == volumeRenderingParams_ )
        return;
    volumeRenderingParams_ = params;
    if ( isVolumeRenderingEnabled() )
        setDirtyFlags( DIRTY_TEXTURE );
}

bool ObjectVoxels::hasVisualRepresentation() const
{
    if ( isVolumeRenderingEnabled() )
        return false;
    return bool( data_.mesh );
}

void ObjectVoxels::setMaxSurfaceVertices( int maxVerts )
{
    if ( maxVerts == maxSurfaceVertices_ )
        return;
    maxSurfaceVertices_ = maxVerts;
    if ( !data_.mesh || data_.mesh->topology.numValidVerts() <= maxSurfaceVertices_ )
        return;
    data_.mesh.reset();
    (void)setIsoValue( isoValue_ ); //TODO: propagate error outside
}

std::shared_ptr<Object> ObjectVoxels::clone() const
{
    auto res = std::make_shared<ObjectVoxels>( ProtectedStruct{}, *this );
    if ( data_.mesh )
        res->data_.mesh = std::make_shared<Mesh>( *data_.mesh );
    if ( vdbVolume_.data )
        res->vdbVolume_.data = MakeFloatGrid( vdbVolume_.data->deepCopy() );
    return res;
}

std::shared_ptr<Object> ObjectVoxels::shallowClone() const
{
    auto res = std::make_shared<ObjectVoxels>( ProtectedStruct{}, *this );
    if ( data_.mesh )
        res->data_.mesh = data_.mesh;
    if ( vdbVolume_.data )
        res->vdbVolume_ = vdbVolume_;
    return res;
}

void ObjectVoxels::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    if ( mask & DIRTY_VOLUME )
    {
        voxelsChangedSignal();
        mask ^= DIRTY_VOLUME;
    }

    ObjectMeshHolder::setDirtyFlags( mask, invalidateCaches );

    if ( invalidateCaches && ( mask & DIRTY_POSITION || mask & DIRTY_FACE ) && data_.mesh )
        data_.mesh->invalidateCaches();
}

size_t ObjectVoxels::activeVoxels() const
{
    if ( !activeVoxels_ )
        activeVoxels_ = vdbVolume_.data ? vdbVolume_.data->activeVoxelCount() : 0;
    return *activeVoxels_;
}

size_t ObjectVoxels::heapBytes() const
{
    return ObjectMeshHolder::heapBytes()
        + vdbVolume_.heapBytes()
        + histogram_.heapBytes()
        + MR::heapBytes( volumeRenderingData_ );
}

void ObjectVoxels::setSerializeFormat( const char * newFormat )
{
    if ( newFormat && *newFormat != '.' )
    {
        assert( false );
        return;
    }
    serializeFormat_ = newFormat;
}

void ObjectVoxels::resetFrontColor()
{
    // cannot implement in the opposite way to keep `setDefaultColors_()` non-virtual
    setDefaultColors_();
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
    {
        std::swap( isoSurfaceChangedSignal, otherVoxels->isoSurfaceChangedSignal );
        std::swap( voxelsChangedSignal, otherVoxels->voxelsChangedSignal );
    }
    else
        assert( false );
}

void ObjectVoxels::updateHistogram_( float min, float max, ProgressCallback cb /*= {}*/ )
{
    MR_TIMER;
    histogram_ = recalculateHistogram( Vector2f{ min, max }, cb );
}

void ObjectVoxels::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectVoxels ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectVoxels ), false );
}

void ObjectVoxels::setDefaultSceneProperties_()
{
    setDefaultColors_();
}

ObjectVoxels::ObjectVoxels()
{
    setDefaultSceneProperties_();
}

void ObjectVoxels::applyScale( float scaleFactor )
{
    vdbVolume_.voxelSize *= scaleFactor;
    reverseVoxelSize_ = { 1 / vdbVolume_.voxelSize.x,1 / vdbVolume_.voxelSize.y,1 / vdbVolume_.voxelSize.z };

    ObjectMeshHolder::applyScale( scaleFactor );
}

void ObjectVoxels::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );
    serializeToJson( vdbVolume_.voxelSize, root["VoxelSize"] );

    const auto activeBounds = getActiveBounds();

    serializeToJson( vdbVolume_.dims, root["Dimensions"] );
    // Min and Max corners are serialized for backward-compatibility
    serializeToJson( activeBounds.min, root["MinCorner"] );
    serializeToJson( activeBounds.max, root["MaxCorner"] );
    //
    serializeToJson( selectedVoxels_, root["SelectionVoxels"] );

    root["IsoValue"] = isoValue_;
    root["DualMarchingCubes"] = dualMarchingCubes_;
    root["Type"].append( ObjectVoxels::TypeName() );
}

Expected<std::future<Expected<void>>> ObjectVoxels::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !vdbVolume_.data )
        return {};

    return std::async( getAsyncLaunchType(),
        [this, filename = std::filesystem::path( path ) += serializeFormat_ ? serializeFormat_ : defaultSerializeVoxelsFormat()] ()
    {
        return MR::VoxelsSave::gridToAnySupportedFormat( vdbVolume_.data, vdbVolume_.dims, filename );
    } );
}

void ObjectVoxels::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );
    if ( root["VoxelSize"].isDouble() )
        vdbVolume_.voxelSize = Vector3f::diagonal( ( float )root["VoxelSize"].asDouble() );
    else
        deserializeFromJson( root["VoxelSize"], vdbVolume_.voxelSize );

    Box3i activeBox;

    deserializeFromJson( root["Dimensions"], vdbVolume_.dims );

    deserializeFromJson( root["MinCorner"], activeBox.min );

    deserializeFromJson( root["MaxCorner"], activeBox.max );

    deserializeFromJson( root["SelectionVoxels"], selectedVoxels_ );

    if ( root["IsoValue"].isNumeric() )
        isoValue_ = root["IsoValue"].asFloat();

    if ( root["DualMarchingCubes"].isBool() )
        dualMarchingCubes_ = root["DualMarchingCubes"].asBool();

    if ( activeBox.valid() && ( activeBox.min != Vector3i() || activeBox.max != vdbVolume_.dims ) )
        setActiveBounds( activeBox );
    else
        (void)setIsoValue( isoValue_ ); // is called automatically in `setActiveBounds`

    if ( root["UseDefaultSceneProperties"].isBool() && root["UseDefaultSceneProperties"].asBool() )
        setDefaultSceneProperties_();
}

Expected<void> ObjectVoxels::deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb )
{
    // in case of raw file, we need to find its full name with suffix
    auto modelPath = pathFromUtf8( utf8string( path ) + ".raw" );
    auto expParams = VoxelsLoad::findRawParameters( modelPath );
    if ( !expParams )
    {
        modelPath = findPathWithExtension( path );
        if ( modelPath.empty() )
            return unexpected( "No voxels file found: " + utf8string( path ) );
    }
    auto res = VoxelsLoad::gridsFromAnySupportedFormat( modelPath, progressCb );
    if ( !res.has_value() )
        return unexpected( res.error() );

    if ( res->empty() )
        return unexpected( "No voxels found in file: " + utf8string( modelPath ) );
    assert( res->size() == 1 );

    construct( ( *res ).front(), vdbVolume_.voxelSize );
    if ( !vdbVolume_.data )
        return unexpected( "No grid loaded" );

    return {};
}

[[nodiscard]] static const char * asString( openvdb::GridClass gc )
{
    switch ( gc )
    {
    case openvdb::GRID_UNKNOWN:
        return "Unknown";
    case openvdb::GRID_LEVEL_SET:
        return "Level Set";
    case openvdb::GRID_FOG_VOLUME:
        return "Fog Volume";
    case openvdb::GRID_STAGGERED:
        return "Staggered";
    default:
        assert( false );
        return "";
    }
}

std::vector<std::string> ObjectVoxels::getInfoLines() const
{
    auto activeBox = getActiveBounds();
    std::vector<std::string> res = ObjectMeshHolder::getInfoLines();
    res.push_back( fmt::format( "dims: ({}, {}, {})", vdbVolume_.dims.x, vdbVolume_.dims.y, vdbVolume_.dims.z ) );
    res.push_back( fmt::format( "voxel size: ({:.3}, {:.3}, {:.3})", vdbVolume_.voxelSize.x, vdbVolume_.voxelSize.y, vdbVolume_.voxelSize.z ) );
    res.push_back( fmt::format( "volume: ({:.3}, {:.3}, {:.3})",
        vdbVolume_.dims.x * vdbVolume_.voxelSize.x,
        vdbVolume_.dims.y * vdbVolume_.voxelSize.y,
        vdbVolume_.dims.z * vdbVolume_.voxelSize.z ) );
    res.push_back( fmt::format( "active box: ({}, {}, {}; {}, {}, {})",
        activeBox.min.x, activeBox.min.y, activeBox.min.z,
        activeBox.max.x, activeBox.max.y, activeBox.max.z ) );
    res.push_back( fmt::format( "min-value: {:.3}", vdbVolume_.min ) );
    res.push_back( fmt::format( "iso-value: {:.3}", isoValue_ ) );
    res.push_back( fmt::format( "max-value: {:.3}", vdbVolume_.max ) );
    res.push_back( dualMarchingCubes_ ? "visual: dual marching cubes" : "visual: standard marching cubes" );

    auto totalVoxels = size_t( vdbVolume_.dims.x ) * vdbVolume_.dims.y * vdbVolume_.dims.z;
    auto activeVoxels = this->activeVoxels();
    res.push_back( "voxels: " + std::to_string( totalVoxels ) );
    if( activeVoxels != totalVoxels )
        res.back() += " / " + std::to_string( activeVoxels ) + " active";
    if ( vdbVolume_.data )
    {
        res.push_back( fmt::format( "background: {:.3}", vdbVolume_.data->background() ) );
        res.push_back( fmt::format( "grid class: {}", asString( vdbVolume_.data->getGridClass() ) ) );
    }

    return res;
}

static std::string sDefaultSerializeVoxelsFormat = ".vdb";

const std::string & defaultSerializeVoxelsFormat()
{
    return sDefaultSerializeVoxelsFormat;
}

void setDefaultSerializeVoxelsFormat( std::string newFormat )
{
    assert( !newFormat.empty() && newFormat[0] == '.' );
    sDefaultSerializeVoxelsFormat = std::move( newFormat );
}

} //namespace MR
