#include "MRVDBConversions.h"

#include "MRVDBFloatGrid.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRTimer.h"
#include "MRVoxelsVolume.h"
#include "MROpenVDB.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRTriMesh.h"
#include "MRVDBProgressInterrupter.h"
#include "MRVoxelsVolumeAccess.h"

namespace MR
{
constexpr float denseVolumeToGridTolerance = 1e-6f;

void convertToVDMMesh( const MeshPart& mp, const AffineXf3f& xf, const Vector3f& voxelSize,
                       std::vector<openvdb::Vec3s>& points, std::vector<openvdb::Vec3I>& tris )
{
    MR_TIMER;
        const auto& pointsRef = mp.mesh.points;
    const auto& topology = mp.mesh.topology;
    points.resize( pointsRef.size() );
    tris.resize( mp.region ? mp.region->count() : topology.numValidFaces() );

    int i = 0;
    VertId v[3];
    for ( FaceId f : topology.getFaceIds( mp.region ) )
    {
        if ( mp.region && !topology.hasFace( f ) )
            continue; // f is in given region but not in mesh topology
        topology.getTriVerts( f, v );
        tris[i++] = openvdb::Vec3I{ ( uint32_t )v[0], ( uint32_t )v[1], ( uint32_t )v[2] };
    }
    i = 0;
    for ( const auto& p0 : pointsRef )
    {
        auto p = xf( p0 );
        points[i][0] = p[0] / voxelSize[0];
        points[i][1] = p[1] / voxelSize[1];
        points[i][2] = p[2] / voxelSize[2];
        ++i;
    }
}

template<typename GridType>
Expected<TriMesh> gridToTriMesh(
    const GridType& grid,
    const GridToMeshSettings & settings )
{
    MR_TIMER;

    if ( !reportProgress( settings.cb, 0.0f ) )
        return unexpectedOperationCanceled();

    openvdb::tools::VolumeToMesh mesher( settings.isoValue, settings.adaptivity, settings.relaxDisorientedTriangles );
    mesher(grid);

    if ( !reportProgress( settings.cb, 0.7f ) )
        return unexpectedOperationCanceled();

    if ( mesher.pointListSize() > settings.maxVertices )
        return unexpected( "Vertices number limit exceeded." );

    // Preallocate the point list
    TriMesh res;
    res.points.resize( mesher.pointListSize() );

    // Copy points
    auto & inPts = mesher.pointList();
    ParallelFor( res.points, [&]( size_t i )
    {
        auto inPt = inPts[i];
        res.points[ VertId{ i } ] = Vector3f{
            inPt.x() * settings.voxelSize.x,
            inPt.y() * settings.voxelSize.y,
            inPt.z() * settings.voxelSize.z };
    } );
    inPts.reset(nullptr);

    if ( !reportProgress( settings.cb, 0.8f ) )
        return unexpectedOperationCanceled();

    auto& polygonPoolList = mesher.polygonPoolList();

    // Preallocate primitive lists
    size_t numQuads = 0, numTriangles = 0;
    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
        openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
        numTriangles += polygons.numTriangles();
        numQuads += polygons.numQuads();
    }

    const size_t tNum = numTriangles + 2 * numQuads;

    if ( tNum > settings.maxFaces )
        return unexpected( "Triangles number limit exceeded." );

    res.tris.reserve( tNum );

    // Copy primitives
    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n)
    {
        openvdb::tools::PolygonPool& polygons = polygonPoolList[n];

        for ( size_t i = 0, I = polygons.numQuads(); i < I; ++i )
        {
            auto quad = polygons.quad(i);

            ThreeVertIds newTri
            {
                VertId( ( int )quad[2] ),
                VertId( ( int )quad[1] ),
                VertId( ( int )quad[0] ),
            };
            res.tris.push_back( newTri );

            newTri =
            {
                VertId( ( int )quad[0] ),
                VertId( ( int )quad[3] ),
                VertId( ( int )quad[2] ),
            };
            res.tris.push_back( newTri );
        }

        for ( size_t i = 0, I = polygons.numTriangles(); i < I; ++i )
        {
            auto tri = polygons.triangle(i);

            ThreeVertIds newTri
            {
                VertId( ( int )tri[2] ),
                VertId( ( int )tri[1] ),
                VertId( ( int )tri[0] )
            };
            res.tris.push_back( newTri );
        }
    }

    if ( !reportProgress( settings.cb, 1.0f ) )
        return unexpectedOperationCanceled();

    return res;
}

FloatGrid meshToLevelSet( const MeshPart& mp, const AffineXf3f& xf,
                          const Vector3f& voxelSize, float surfaceOffset,
                          ProgressCallback cb )
{
    if ( surfaceOffset <= 0.0f )
    {
        assert( false );
        return {};
    }
    MR_TIMER;
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> tris;

    convertToVDMMesh( mp, xf, voxelSize, points, tris );

    openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform();
    ProgressInterrupter interrupter( cb );
    auto resGrid = MakeFloatGrid( openvdb::tools::meshToLevelSet<openvdb::FloatGrid, ProgressInterrupter>
        ( interrupter, *xform, points, tris, surfaceOffset ) );
    if ( interrupter.getWasInterrupted() )
        return {};
    return resGrid;
}

FloatGrid meshToDistanceField( const MeshPart& mp, const AffineXf3f& xf,
    const Vector3f& voxelSize, float surfaceOffset /*= 3 */,
    ProgressCallback cb )
{
    if ( surfaceOffset <= 0.0f )
    {
        assert( false );
        return {};
    }
    MR_TIMER;
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> tris;

    convertToVDMMesh( mp, xf, voxelSize, points, tris );

    openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform();
    ProgressInterrupter interrupter( cb );

    auto resGrid = MakeFloatGrid( openvdb::tools::meshToUnsignedDistanceField<openvdb::FloatGrid, ProgressInterrupter>
        ( interrupter, *xform, points, tris, {}, surfaceOffset ) );

    if ( interrupter.getWasInterrupted() )
        return {};
    return resGrid;
}

void evalGridMinMax( const FloatGrid& grid, float& min, float& max )
{
    if ( !grid )
        return;
    MR_TIMER;
#if (OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER >= 9 && (OPENVDB_LIBRARY_MINOR_VERSION_NUMBER >= 1 || OPENVDB_LIBRARY_PATCH_VERSION_NUMBER >= 1)) || \
    (OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER >= 10)
    auto minMax = openvdb::tools::minMax( grid->tree() );
    min = minMax.min();
    max = minMax.max();
#else
    grid->evalMinMax( min, max );
#endif
}

Expected<VdbVolume> meshToDistanceVdbVolume( const MeshPart& mp, const MeshToVolumeParams& params /*= {} */ )
{
    if ( params.type == MeshToVolumeParams::Type::Signed && !mp.mesh.topology.isClosed( mp.region ) )
        return unexpected( "Only closed mesh can be converted to signed volume" );
    MR_TIMER;

    FloatGrid grid;
    if ( params.type == MeshToVolumeParams::Type::Signed )
        grid = meshToLevelSet( mp, params.worldXf, params.voxelSize, params.surfaceOffset, params.cb );
    else
        grid = meshToDistanceField( mp, params.worldXf, params.voxelSize, params.surfaceOffset, params.cb );

    if ( !grid )
        return unexpectedOperationCanceled();

    // to get proper normal orientation both for signed and unsigned cases
    grid->setGridClass( openvdb::GRID_LEVEL_SET );

    VdbVolume res;
    res.data = grid;
    evalGridMinMax( grid, res.min, res.max );
    auto dim = grid->evalActiveVoxelBoundingBox().extents();
    res.dims = Vector3i( dim.x(), dim.y(), dim.z() );
    res.voxelSize = params.voxelSize;

    return res;
}

Expected<VdbVolume> meshToVolume( const MeshPart& mp, const MeshToVolumeParams& cParams /*= {} */ )
{
    MR_TIMER;

    auto shift = AffineXf3f::translation( mp.mesh.computeBoundingBox( mp.region, &cParams.worldXf ).min
        - cParams.surfaceOffset * cParams.voxelSize );
    if ( cParams.outXf )
        *cParams.outXf = shift;

    auto params = cParams;
    params.worldXf = shift.inverse() * cParams.worldXf;
    return meshToDistanceVdbVolume( mp, params );
}

VdbVolume floatGridToVdbVolume( FloatGrid grid )
{
    if ( !grid )
        return {};
    MR_TIMER;
    VdbVolume res;
    evalGridMinMax( grid, res.min, res.max );
    auto dim = grid->evalActiveVoxelDim();
    res.dims = Vector3i( dim.x(), dim.y(), dim.z() );
    res.data = std::move( grid );
    return res;
}

template <>
void putSimpleVolumeInDenseGrid(
        openvdb::FloatGrid& grid,
        const Vector3i& minCoord, const SimpleVolume& simpleVolume, ProgressCallback cb
    )
{
    MR_TIMER;
    if ( cb )
        cb( 0.0f );
    openvdb::math::Coord dimsCoord( simpleVolume.dims.x, simpleVolume.dims.y, simpleVolume.dims.z );
    openvdb::math::CoordBBox denseBBox( toVdb( minCoord ), toVdb( minCoord ) + dimsCoord.offsetBy( -1 ) );
    openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense( denseBBox, const_cast< float* >( simpleVolume.data.data() ) );
    if ( cb )
        cb( 0.5f );
    openvdb::tools::copyFromDense( dense, grid, denseVolumeToGridTolerance );
    if ( cb )
        cb( 1.f );
}

template <>
void putSimpleVolumeInDenseGrid(
        FloatGrid& grid,
        const Vector3i& minCoord, const SimpleVolume& simpleVolume, ProgressCallback cb
    )
{
    openvdb::FloatGrid& gridRef = *grid;
    putSimpleVolumeInDenseGrid( gridRef, minCoord, simpleVolume, cb );
}

template <typename VolumeType>
void putVolumeInDenseGrid(
        openvdb::FloatGrid::Accessor& gridAccessor,
        const Vector3i& minCoord, const VolumeType& volume, ProgressCallback cb )
{
    MR_TIMER;
    if ( cb )
        cb( 0.0f );

    VoxelsVolumeAccessor<VolumeType> volumeAccessor( volume );

    for ( int z = 0; z < volume.dims.z; ++z )
    {
        if ( !reportProgress( cb, ( float )z / ( float )volume.dims.z ) )
            return;
        for ( int y = 0; y < volume.dims.y; ++y )
        {
            for ( int x = 0; x < volume.dims.x; ++x )
            {
                auto loc = Vector3i{ x, y, z };
                auto coord = toVdb( minCoord + loc );
                gridAccessor.setValue( coord, volumeAccessor.get( loc ) );
            }
        }
    }
}

template <>
void putSimpleVolumeInDenseGrid(
        openvdb::FloatGrid::Accessor& gridAccessor,
        const Vector3i& minCoord, const SimpleVolume& simpleVolume, ProgressCallback cb
    )
{
    putVolumeInDenseGrid( gridAccessor, minCoord, simpleVolume, cb );
}

FloatGrid simpleVolumeToDenseGrid( const SimpleVolume& simpleVolume,
                                   float background,
                                   ProgressCallback cb )
{
    MR_TIMER;
    std::shared_ptr<openvdb::FloatGrid> grid = std::make_shared<openvdb::FloatGrid>( FLT_MAX );
    putSimpleVolumeInDenseGrid( *grid, { 0, 0, 0 }, simpleVolume, cb );
    openvdb::tools::changeBackground( grid->tree(), background );
    return MakeFloatGrid( std::move( grid ) );
}

VdbVolume simpleVolumeToVdbVolume( const SimpleVolumeMinMax& simpleVolume, ProgressCallback cb /*= {} */ )
{
    VdbVolume res;
    res.data = simpleVolumeToDenseGrid( simpleVolume, simpleVolume.min, cb );
    res.dims = simpleVolume.dims;
    res.voxelSize = simpleVolume.voxelSize;
    res.min = simpleVolume.min;
    res.max = simpleVolume.max;
    return res;
}

VdbVolume functionVolumeToVdbVolume( const FunctionVolume& functoinVolume, ProgressCallback cb /*= {} */ )
{
    MR_TIMER;
    VdbVolume res;
    std::shared_ptr<openvdb::FloatGrid> grid = std::make_shared<openvdb::FloatGrid>( FLT_MAX );
    auto gridAccessor = grid->getAccessor();
    putVolumeInDenseGrid( gridAccessor, { 0, 0, 0 }, functoinVolume, cb );
    auto minMax = openvdb::tools::minMax( grid->tree() );
    res.min = minMax.min();
    res.max = minMax.max();
    openvdb::tools::changeBackground( grid->tree(), res.min );
    res.data = MakeFloatGrid( std::move( grid ) );
    res.dims = functoinVolume.dims;
    res.voxelSize = functoinVolume.voxelSize;

    return res;
}

// make VoxelsVolume (e.g. SimpleVolume or SimpleVolumeU16) from VdbVolume
// if VoxelsVolume values type is integral, performs mapping from the sourceScale to
// nonnegative range of target type
template<typename T, bool Norm>
Expected<VoxelsVolumeMinMax<Vector<T,VoxelId>>> vdbVolumeToSimpleVolumeImpl(
    const VdbVolume& vdbVolume, const Box3i& activeBox = Box3i(), std::optional<MinMaxf> maybeSourceScale = {}, ProgressCallback cb = {} )
{
    MR_TIMER;
    constexpr bool isFloat = std::is_same_v<float, T> || std::is_same_v<double, T> || std::is_same_v<long double, T>;

    VoxelsVolumeMinMax<Vector<T,VoxelId>> res;

    res.dims = !activeBox.valid() ? vdbVolume.dims : activeBox.size();
    Vector3i org = activeBox.valid() ? activeBox.min : Vector3i{};
    res.voxelSize = vdbVolume.voxelSize;
    [[maybe_unused]] const auto sourceScale = maybeSourceScale.value_or( MinMaxf{ vdbVolume.min, vdbVolume.max } );
    float targetMin = sourceScale.min, targetMax = sourceScale.max;
    if constexpr ( isFloat )
    {
        if constexpr ( Norm )
        {
            targetMin = 0;
            targetMax = 1;
        }
        else
        {
            targetMin = vdbVolume.min;
            targetMax = vdbVolume.max;
        }
    }
    else
    {
        targetMin = 0;
        targetMax = std::numeric_limits<T>::max();
    }
    [[maybe_unused]] const float k = ( targetMax - targetMin ) / ( sourceScale.max - sourceScale.min );
    res.min = T( k * ( vdbVolume.min - sourceScale.min ) + targetMin );
    res.max = T( k * ( vdbVolume.max - sourceScale.min ) + targetMin );

    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    if ( !vdbVolume.data )
        return res;

    tbb::enumerable_thread_specific accessorPerThread( vdbVolume.data->getConstAccessor() );
    if ( !ParallelFor( 0_vox, indexer.endId(), [&]( VoxelId i )
    {
        auto& accessor = accessorPerThread.local();
        auto coord = indexer.toPos( i );
        float value = accessor.getValue( openvdb::Coord( coord.x + org.x, coord.y + org.y, coord.z + org.z ) );
        if constexpr ( isFloat && !Norm )
            res.data[i] = T( value );
        else
            res.data[i] = T( std::clamp( ( value - sourceScale.min ) * k + targetMin, targetMin, targetMax ) );
    }, cb ) )
        return unexpectedOperationCanceled();
    return res;
}

Expected<SimpleVolumeMinMax> vdbVolumeToSimpleVolume( const VdbVolume& vdbVolume, const Box3i& activeBox, ProgressCallback cb )
{
    return vdbVolumeToSimpleVolumeImpl<float, false>( vdbVolume, activeBox, {}, cb );
}

Expected<SimpleVolumeMinMax> vdbVolumeToSimpleVolumeNorm( const VdbVolume& vdbVolume, const Box3i& activeBox /*= Box3i()*/,
                                                          std::optional<MinMaxf> sourceScale, ProgressCallback cb /*= {} */ )
{
    return vdbVolumeToSimpleVolumeImpl<float, true>( vdbVolume, activeBox, sourceScale, cb );
}

Expected<SimpleVolumeMinMaxU16> vdbVolumeToSimpleVolumeU16( const VdbVolume& vdbVolume, const Box3i& activeBox,
                                                            std::optional<MinMaxf> sourceScale, ProgressCallback cb )
{
    return vdbVolumeToSimpleVolumeImpl<uint16_t, true>( vdbVolume, activeBox, sourceScale, cb );
}

Expected<Mesh> gridToMesh( const FloatGrid& grid, const GridToMeshSettings & settings )
{
    MR_TIMER;
    if ( !reportProgress( settings.cb, 0.0f ) )
        return unexpectedOperationCanceled();

    auto s = settings;
    s.cb = subprogress( settings.cb, 0.0f, 0.2f );
    auto expTriMesh = gridToTriMesh( *grid, s );
    if ( !expTriMesh )
        return unexpected( std::move( expTriMesh.error() ) );

    if ( !reportProgress( settings.cb, 0.2f ) )
        return unexpectedOperationCanceled();

    Mesh res = Mesh::fromTriMesh( std::move( *expTriMesh ), {}, subprogress( settings.cb, 0.2f, 1.0f ) );
    if ( !reportProgress( settings.cb, 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

Expected<Mesh> gridToMesh( FloatGrid&& grid, const GridToMeshSettings & settings )
{
    MR_TIMER;
    if ( !reportProgress( settings.cb, 0.0f ) )
        return unexpectedOperationCanceled();

    auto s = settings;
    s.cb = subprogress( settings.cb, 0.0f, 0.2f );
    auto expTriMesh = gridToTriMesh( *grid, s );
    if ( !expTriMesh )
        return unexpected( std::move( expTriMesh.error() ) );
    grid.reset(); // free grid's memory

    if ( !reportProgress( settings.cb, 0.2f ) )
        return unexpectedOperationCanceled();

    Mesh res = Mesh::fromTriMesh( std::move( *expTriMesh ), {}, subprogress( settings.cb, 0.2f, 1.0f ) );
    if ( !reportProgress( settings.cb, 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

Expected<void> makeSignedByWindingNumber( FloatGrid& grid, const Vector3f& voxelSize, const Mesh& refMesh, const MakeSignedByWindingNumberSettings & settings )
{
    MR_TIMER;

    auto activeBox = grid->evalActiveVoxelBoundingBox();
    // make dense topology tree to copy its nodes topology to original grid
    std::unique_ptr<openvdb::TopologyTree> topologyTree = std::make_unique<openvdb::TopologyTree>();
    // make it dense
    topologyTree->denseFill( activeBox, {} );
    grid->tree().topologyUnion( *topologyTree ); // after this all voxels should be active and trivial parallelism is ok
    // free topology tree
    topologyTree.reset();

    auto minCoord = activeBox.min();
    auto dims = activeBox.dim();
    VolumeIndexer indexer( Vector3i( dims.x(), dims.y(), dims.z() ) );
    const size_t volume = activeBox.volume();

    auto fwn = settings.fwn ? settings.fwn : std::make_shared<FastWindingNumber>( refMesh );

    const auto gridToMeshXf = settings.meshToGridXf.inverse()
        * AffineXf3f::linear( Matrix3f::scale( voxelSize ) )
        * AffineXf3f::translation( Vector3f{ fromVdb( minCoord ) } );

    tbb::enumerable_thread_specific<openvdb::FloatGrid::Accessor> perThreadAccessor( grid->getAccessor() );
    auto updateGrid = [&] ( VoxelId vox, float windVal )
    {
        auto & accessor = perThreadAccessor.local();

        auto pos = indexer.toPos( vox );
        auto coord = minCoord;
        for ( int j = 0; j < 3; ++j )
            coord[j] += pos[j];

        if ( windVal > settings.windingNumberThreshold )
        {
            accessor.modifyValue( coord, [] ( float& val )
            {
                val = -val;
            } );
        }
    };

    if ( auto fwnByParts = std::dynamic_pointer_cast<IFastWindingNumberByParts>( fwn ) )
    {
        auto func = [&] ( std::vector<float>&& vals, const Vector3i&, int zOffset ) -> Expected<void>
        {
            const auto offset = indexer.sizeXY() * zOffset;
            ParallelFor( size_t( 0 ), vals.size(), [&] ( size_t i )
            {
                updateGrid( VoxelId( i + offset ), vals[i] );
            } );
            return {};
        };
        return
            fwnByParts->calcFromGridByParts( func, indexer.dims(), gridToMeshXf, settings.windingNumberBeta, 0, settings.progress )
            .transform( [&]
            {
                grid->pruneGrid( 0.f );
            } );
    }

    std::vector<float> windVals;
    if ( auto res = fwn->calcFromGrid( windVals,
        Vector3i{ dims.x(),  dims.y(), dims.z() },
        gridToMeshXf, settings.windingNumberBeta, subprogress( settings.progress, 0.0f, 0.8f ) ); !res )
    {
        return res;
    }

    if ( !ParallelFor( size_t( 0 ), volume, [&]( size_t i )
        {
            updateGrid( VoxelId( i ), windVals[i] );
        }, subprogress( settings.progress, 0.8f, 1.0f ) ) )
    {
        return unexpectedOperationCanceled();
    }

    grid->pruneGrid( 0.0f );
    return {};
}

static FloatGrid meshToUnsignedDistanceField_(
    const std::vector<openvdb::Vec3s> & points, std::vector<openvdb::Vec3I> & tris, const std::vector<openvdb::Vec4I> & quads,
    float surfaceOffset, const ProgressCallback & cb )
{
    assert ( surfaceOffset > 0 );
    MR_TIMER;
    openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform();
    ProgressInterrupter interrupter( cb );
    auto resGrid = MakeFloatGrid( openvdb::tools::meshToUnsignedDistanceField<openvdb::FloatGrid, ProgressInterrupter>
        ( interrupter, *xform, points, tris, quads, surfaceOffset ) );
    if ( interrupter.getWasInterrupted() )
        return {};
    return resGrid;
}

static FloatGrid meshToLevelSet_(
    const std::vector<openvdb::Vec3s> & points, std::vector<openvdb::Vec3I> & tris, const std::vector<openvdb::Vec4I> & quads,
    float surfaceOffset, const ProgressCallback & cb )
{
    assert ( surfaceOffset > 0 );
    MR_TIMER;
    openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform();
    ProgressInterrupter interrupter( cb );
    auto resGrid = MakeFloatGrid( openvdb::tools::meshToLevelSet<openvdb::FloatGrid, ProgressInterrupter>
        ( interrupter, *xform, points, tris, quads, surfaceOffset ) );
    if ( interrupter.getWasInterrupted() )
        return {};
    return resGrid;
}

Expected<Mesh> doubleOffsetVdb( const MeshPart& mp, const DoubleOffsetSettings & settings )
{
    MR_TIMER;

    auto offsetInVoxelsA = settings.offsetA / settings.voxelSize;
    auto offsetInVoxelsB = settings.offsetB / settings.voxelSize;

    if ( !reportProgress( settings.progress, 0.0f ) )
        return unexpectedOperationCanceled();

    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> tris;
    convertToVDMMesh( mp, AffineXf3f(), Vector3f::diagonal( settings.voxelSize ), points, tris );

    if ( !reportProgress( settings.progress, 0.1f ) )
        return unexpectedOperationCanceled();

    const bool needSignUpdate = !mp.mesh.topology.isClosed( mp.region );

    auto sp = subprogress( settings.progress, 0.1f, needSignUpdate ? 0.2f : 0.3f );
    auto grid = needSignUpdate ?
        meshToUnsignedDistanceField_( points, tris, {}, std::abs( offsetInVoxelsA ) + 1, sp ) :
        meshToLevelSet_( points, tris, {}, std::abs( offsetInVoxelsA ) + 1, sp );

    if ( !grid || !reportProgress( sp, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( needSignUpdate )
    {
        auto signRes = makeSignedByWindingNumber( grid, Vector3f::diagonal( settings.voxelSize ), mp.mesh,
        {
            .fwn = settings.fwn,
            .windingNumberThreshold = settings.windingNumberThreshold,
            .windingNumberBeta = settings.windingNumberBeta,
            .progress = subprogress( settings.progress, 0.2f, 0.3f )
        } );
        if ( !signRes.has_value() )
            return unexpected( signRes.error() );
    }

    std::vector<openvdb::Vec4I> quads;
    {
        Timer t( "volumeToMesh" );
        openvdb::tools::volumeToMesh( *grid, points, tris, quads, offsetInVoxelsA, settings.adaptivity );
    }

    if ( !reportProgress( settings.progress, 0.5f ) )
        return unexpectedOperationCanceled();
    sp = subprogress( settings.progress, 0.5f, 0.9f );

    grid = meshToLevelSet_( points, tris, quads, std::abs( offsetInVoxelsB ) + 1, sp );
    if ( !grid || !reportProgress( sp, 1.0f ) )
        return unexpectedOperationCanceled();

    auto expTriMesh = gridToTriMesh( *grid, GridToMeshSettings{
        .voxelSize = Vector3f::diagonal( settings.voxelSize ),
        .isoValue = offsetInVoxelsB,
        .adaptivity = settings.adaptivity,
        .cb = subprogress( settings.progress, 0.9f, 0.95f )
    } );

    Mesh res = Mesh::fromTriMesh( std::move( *expTriMesh ) );
    if ( !reportProgress( settings.progress, 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

} //namespace MR
