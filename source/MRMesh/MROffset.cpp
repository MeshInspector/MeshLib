#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MROffset.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRVDBConversions.h"
#include "MRTimer.h"
#include "MRPolyline.h"
#include "MRMeshFillHole.h"
#include "MRRegionBoundary.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MROpenvdb.h"
#include "MRTriMath.h"

namespace
{
constexpr float autoVoxelNumber = 5e6f;
}

namespace MR
{

tl::expected<Mesh, std::string> offsetMesh( const MeshPart & mp, float offset, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER

    float voxelSize = params.voxelSize;
    // Compute voxel size if needed
    if ( voxelSize <= 0.0f )
    {
        auto bb = mp.mesh.computeBoundingBox( mp.region );
        auto vol = bb.volume();
        voxelSize = std::cbrt( vol / autoVoxelNumber );
    }

    bool useShell = params.type == OffsetParameters::Type::Shell;
    if ( !findRegionBoundary( mp.mesh.topology, mp.region ).empty() && !useShell )
    {
        spdlog::warn( "Cannot use offset for non-closed meshes, using shell instead." );
        useShell = true;
    }

    if ( useShell )
        offset = std::abs( offset );

    auto offsetInVoxels = offset / voxelSize;

    auto voxelSizeVector = Vector3f::diagonal( voxelSize );
    // Make grid

    // CREATE MESH 1
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> tris;

    convertToVDMMesh( mp, AffineXf3f(), voxelSizeVector, points, tris );

    typename openvdb::FloatGrid::ConstPtr refGrid;
    using IntGridT = typename openvdb::FloatGrid::template ValueConverter<openvdb::Int32>::Type;
    typename IntGridT::Ptr indexGrid; // replace

    openvdb::tools::MeshToVoxelEdgeData edgeData;

    openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec3I>
        mesh( points, tris );

    indexGrid.reset( new IntGridT( 0 ) );

    openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform();
    Interrupter interrupter( params.callBack );

    auto grid = openvdb::tools::meshToVolume<openvdb::FloatGrid>( interrupter,
        mesh, *xform, std::abs( offsetInVoxels ) + 1, std::abs( offsetInVoxels ) + 1, 0, indexGrid.get() );


    std::vector<openvdb::Vec4I> quads( tris.size(), openvdb::Vec4I( openvdb::util::INVALID_IDX ) );
    for ( int i = 0; i < quads.size(); ++i )
        *( openvdb::Vec3I* )( &quads[i] ) = tris[i];

    edgeData.convert( points, quads );



    /// CREATE ADAPTIVITY MASK

    openvdb::tools::VolumeToMesh mesher( offsetInVoxels, params.adaptivity );

    using TreeType = typename openvdb::FloatGrid::TreeType;
    using ValueType = typename openvdb::FloatGrid::ValueType;

    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;
    typename BoolTreeType::Ptr maskTree;

    maskTree = typename BoolTreeType::Ptr( new BoolTreeType( false ) );
    maskTree->topologyUnion( indexGrid->tree() );
    openvdb::tree::LeafManager<BoolTreeType> maskLeafs( *maskTree );

    float VAR_edgeTolerance = 0.0f; // [0,1]
    auto range = maskLeafs.getRange();

    using IndexAccessorType = typename openvdb::tree::ValueAccessor<const IntGridT::TreeType>;
    IndexAccessorType idxAcc( indexGrid->tree() );

    Vector3f tmpN, normal;
    int tmpIdx;

    openvdb::Coord ijk, nijk;
    typename BoolTreeType::LeafNodeType::ValueOnIter iter;
    
    auto primNorm = [&] ( int index )
    {
        const auto& tri = tris[index];
        const auto& pa = points[tri[0]];
        Vector3f a = Vector3f( pa[0], pa[1], pa[2] );

        const auto& pb = points[tri[1]];
        Vector3f b = Vector3f( pb[0], pb[1], pb[2] );

        const auto& pc = points[tri[2]];
        Vector3f c = Vector3f( pc[0], pc[1], pc[2] );
        return dirDblArea( a, b, c ).normalized();
    };
    for ( size_t n = range.begin(); n < range.end(); ++n )
    {
        iter = maskLeafs.leaf( n ).beginValueOn();
        for ( ; iter; ++iter )
        {
            ijk = iter.getCoord();

            bool edgeVoxel = false;

            int idx = idxAcc.getValue( ijk );

            normal = primNorm( idx );

            for ( size_t i = 0; i < 18; ++i )
            {
                nijk = ijk + openvdb::util::COORD_OFFSETS[i];
                if ( idxAcc.probeValue( nijk, tmpIdx ) && tmpIdx != idx )
                {
                    tmpN = primNorm( tmpIdx );

                    if ( dot( normal, tmpN ) < VAR_edgeTolerance )
                    {
                        edgeVoxel = true;
                        break;
                    }
                }
            }

            if ( !edgeVoxel ) iter.setValueOff();
        }
    }

    openvdb::tools::pruneInactive( *maskTree );

    openvdb::tools::dilateActiveValues( *maskTree, 2,
        openvdb::tools::NN_FACE, openvdb::tools::IGNORE_TILES );

    mesher.setAdaptivityMask( maskTree );


    float VAR_internalAdaptivity = 0.0f; // [0,1]
    mesher.setRefGrid( grid, VAR_internalAdaptivity );


    mesher( *grid );

    using BoolAccessor = openvdb::tree::ValueAccessor<const openvdb::BoolTree>;
    std::unique_ptr<BoolAccessor> maskAcc = std::make_unique<BoolAccessor>( *maskTree.get() );
    // CREATE MESH 2

    openvdb::tools::MeshToVoxelEdgeData::Accessor acc = edgeData.getAccessor();
    std::vector<openvdb::Vec3d> pointsNeigh, normalsNeigh;
    std::vector<openvdb::Index32> primitivesNeigh;
    pointsNeigh.reserve( 12 );
    normalsNeigh.reserve( 12 );
    primitivesNeigh.reserve( 12 );
    auto& ptList = mesher.pointList();
    for ( int i = 0; i < mesher.pointListSize(); ++i )
    {
        auto& pt = ptList[i];
        openvdb::Coord ijk2;
        ijk2[0] = int( std::floor( pt[0] ) );
        ijk2[1] = int( std::floor( pt[1] ) );
        ijk2[2] = int( std::floor( pt[2] ) );
        if ( maskAcc && !maskAcc->isValueOn( ijk2 ) )
            continue;

        pointsNeigh.clear();
        normalsNeigh.clear();
        primitivesNeigh.clear();

        // get voxel-edge intersections
        edgeData.getEdgeData( acc, ijk2, pointsNeigh, primitivesNeigh );
        for ( int p = 0; p < primitivesNeigh.size(); ++p )
        {
            auto tmpNorm = primNorm( primitivesNeigh[p] );
            normalsNeigh.push_back( openvdb::Vec3d( tmpNorm.x, tmpNorm.y, tmpNorm.z ) );
        }
        if ( pointsNeigh.size() > 1 )
            pt = openvdb::tools::findFeaturePoint( pointsNeigh, normalsNeigh );
    }


    // CREATE RESULT MESH
    VertCoords resPoints( mesher.pointListSize() );
    Triangulation resTriangulation;

    // Copy points
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, resPoints.size() ), [&] ( const tbb::blocked_range<size_t>& rangePt )
    {
        for ( auto i = rangePt.begin(); i < rangePt.end(); ++i )
        {
            auto inPt = ptList[i];
            resPoints[VertId{ i }] = Vector3f{
                inPt.x()* voxelSizeVector.x,
                inPt.y()* voxelSizeVector.y,
                inPt.z()* voxelSizeVector.z };
        }
    } );
    //inPts.reset( nullptr );

    auto& polygonPoolList = mesher.polygonPoolList();

    // Preallocate primitive lists
    size_t numQuads = 0, numTriangles = 0;
    for ( size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n )
    {
        openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
        numTriangles += polygons.numTriangles();
        numQuads += polygons.numQuads();
    }

    const size_t tNum = numTriangles + 2 * numQuads;
    resTriangulation.clear();
    resTriangulation.reserve( tNum );

    // Copy primitives
    for ( size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n )
    {
        openvdb::tools::PolygonPool& polygons = polygonPoolList[n];

        for ( size_t i = 0, I = polygons.numQuads(); i < I; ++i )
        {
            auto quad = polygons.quad( i );

            ThreeVertIds newTri
            {
                VertId( ( int )quad[2] ),
                VertId( ( int )quad[1] ),
                VertId( ( int )quad[0] ),
            };
            resTriangulation.push_back( newTri );

            newTri =
            {
                VertId( ( int )quad[0] ),
                VertId( ( int )quad[3] ),
                VertId( ( int )quad[2] ),
            };
            resTriangulation.push_back( newTri );
        }

        for ( size_t i = 0, I = polygons.numTriangles(); i < I; ++i )
        {
            auto tri = polygons.triangle( i );

            ThreeVertIds newTri
            {
                VertId( ( int )tri[2] ),
                VertId( ( int )tri[1] ),
                VertId( ( int )tri[0] )
            };
            resTriangulation.push_back( newTri );
        }
    }


    return Mesh::fromTriangles( std::move( resPoints ), resTriangulation );



    //auto grid = ( !useShell ) ?
    //    // Make level set grid if it is closed
    //    meshToLevelSet( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 1,
    //                    params.callBack ?
    //                    [params]( float p )
    //{
    //    return params.callBack( p * 0.5f );
    //} : ProgressCallback{} ) :
    //    // Make distance field grid if it is not closed
    //    meshToDistanceField( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 1,
    //                    params.callBack ?
    //                         [params]( float p )
    //{
    //    return params.callBack( p * 0.5f );
    //} : ProgressCallback{} );
    //
    //if ( !grid )
    //    return tl::make_unexpected( "Operation was canceled." );
    //
    //// Make offset mesh
    //auto newMesh = gridToMesh( std::move( grid ), voxelSizeVector, offsetInVoxels, params.adaptivity, params.callBack ?
    //                         [params]( float p )
    //{
    //    return params.callBack( 0.5f + p * 0.5f );
    //} : ProgressCallback{} );
    //
    //if ( !newMesh.has_value() )
    //    return tl::make_unexpected( "Operation was canceled." );
    //
    //// For not closed meshes orientation is flipped on back conversion
    //if ( useShell )
    //    newMesh->topology.flipOrientation();

    //return newMesh;
}

tl::expected<Mesh, std::string> doubleOffsetMesh( const MeshPart& mp, float offsetA, float offsetB, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER
    if ( !findRegionBoundary( mp.mesh.topology, mp.region ).empty() )
    {
        spdlog::error( "Only closed meshes allowed for double offset." );
        return tl::make_unexpected( "Only closed meshes allowed for double offset." );
    }
    if ( params.type == OffsetParameters::Type::Shell )
    {
        spdlog::warn( "Cannot use shell for double offset, using offset mode instead." );
    }
    return levelSetDoubleConvertion( mp, AffineXf3f(), params.voxelSize, offsetA, offsetB, params.adaptivity, params.callBack );
}

tl::expected<Mesh, std::string> offsetPolyline( const Polyline3& polyline, float offset, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER;

    Mesh mesh;
    auto contours = polyline.topology.convertToContours<Vector3f>(
        [&points = polyline.points]( VertId v )
    {
        return points[v];
    } );

    std::vector<EdgeId> newHoles;
    newHoles.reserve( contours.size() );
    for ( auto& cont : contours )
    {
        if ( cont[0] != cont.back() )
            cont.insert( cont.end(), cont.rbegin(), cont.rend() );
        newHoles.push_back( mesh.addSeparateEdgeLoop( cont ) );
    }

    for ( auto h : newHoles )
        makeDegenerateBandAroundHole( mesh, h );

    return offsetMesh( mesh, offset, params );
}

}
#endif
