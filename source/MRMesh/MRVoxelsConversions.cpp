#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRVoxelsConversions.h"
#include "MRMesh.h"
#include "MRVolumeIndexer.h"
#include "MRIntersectionPrecomputes.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MRMeshBuilder.h"
#include "MRFloatGrid.h"
#include "MRTimer.h"
#include "MRFastWindingNumber.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MROpenvdb.h"
#include <limits>
#include <optional>
#include <thread>

namespace MR
{

class MinMaxCalc
{
public:
    MinMaxCalc( const std::vector<float>& vec )
    : vec_( vec )
    {}

    MinMaxCalc(const MinMaxCalc& other, tbb::split)
    : min_(other.min_)
    , max_(other.max_)
    ,vec_( other.vec_)
    {}

    void operator()( const tbb::blocked_range<size_t>& r )
    {
        for ( auto i = r.begin(); i < r.end(); ++i )
        {
            if ( vec_[i] < min_ )
                min_ = vec_[i];

            if ( vec_[i] > max_ )
                max_ = vec_[i];
        }
    }

    void join( const MinMaxCalc& other )
    {
        min_ = std::min( min_, other.min_ );
        max_ = std::max( max_, other.max_ );
    }

    float min() { return min_; }
    float max() { return max_; }

private:
    float min_{ FLT_MAX };
    float max_{ -FLT_MAX };
    const std::vector<float>& vec_;
};

std::optional<SimpleVolume> meshToSimpleVolume( const Mesh& mesh, const MeshToSimpleVolumeParams& params /*= {} */ )
{
    MR_TIMER
    SimpleVolume res;
    auto transposedBasis = params.basis.A.transposed();
    for ( int i = 0; i < 3; ++i )
        res.voxelSize[i] = transposedBasis[i].length();
    res.dims = params.dimensions;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    // used in Winding rule mode
    const IntersectionPrecomputes<double> precomputedInter( Vector3d::plusX() );
    
    if ( params.signMode == SignDetectionMode::HoleWindingRule )
    {
        auto fwn = params.fwn;
        if ( !fwn )
            fwn = std::make_shared<FastWindingNumber>( mesh );

        constexpr float beta = 2;
        fwn->calcFromGridWithDistances( res.data, res.dims, Vector3f::diagonal( 0.5f ), Vector3f::diagonal( 1.0f ), params.basis, beta, params.maxDistSq, params.minDistSq );
        MinMaxCalc minMaxCalc( res.data );
        tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, res.data.size() ), minMaxCalc );
        res.min = minMaxCalc.min();
        res.max = minMaxCalc.max();
        return res;
    }

    std::atomic<bool> keepGoing{ true };
    auto mainThreadId = std::this_thread::get_id();
    tbb::enumerable_thread_specific<std::pair<float, float>> minMax( std::pair<float, float>{ FLT_MAX, -FLT_MAX } );

    auto core = [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            if ( params.cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            auto coord = Vector3f( indexer.toPos( VoxelId( i ) ) ) + Vector3f::diagonal( 0.5f );
            auto voxelCenter = params.basis.b + params.basis.A * coord;
            float dist{ 0.0f };
            if ( params.signMode != SignDetectionMode::ProjectionNormal )
                dist = std::sqrt( findProjection( voxelCenter, mesh, params.maxDistSq, nullptr, params.minDistSq ).distSq );
            else
            {
                auto s = findSignedDistance( voxelCenter, mesh, params.maxDistSq, params.minDistSq );
                dist = s ? s->dist : std::numeric_limits<float>::quiet_NaN();
            }

            if ( !std::isnan( dist ) )
            {
                bool changeSign = false;
                if ( params.signMode == SignDetectionMode::WindingRule )
                {
                    int numInters = 0;
                    rayMeshIntersectAll( mesh, Line3d( Vector3d( voxelCenter ), Vector3d::plusX() ),
                        [&numInters] ( const MeshIntersectionResult& ) mutable
                    {
                        ++numInters;
                        return true;
                    } );
                    changeSign = numInters % 2 == 1; // inside
                }
                if ( changeSign )
                    dist = -dist;
                auto& localMinMax = minMax.local();
                if ( dist < localMinMax.first )
                    localMinMax.first = dist;
                if ( dist > localMinMax.second )
                    localMinMax.second = dist;
            }
            res.data[i] = dist;
            if ( params.cb && ( ( i % 1024 ) == 0 ) && std::this_thread::get_id() == mainThreadId )
            {
                if ( !params.cb( float( i ) / float( range.size() ) ) )
                    keepGoing.store( false, std::memory_order_relaxed );
            }
        }
    };

    if ( params.cb )
        // static partitioner is slower but is necessary for smooth progress reporting
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, indexer.size() ), core, tbb::static_partitioner() );
    else
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, indexer.size() ), core );

    if ( params.cb && !keepGoing )
        return {};
    for ( const auto& [min, max] : minMax )
    {
        if ( min < res.min )
            res.min = min;
        if ( max > res.max )
            res.max = max;
    }
    return res;
}

namespace MarchingCubesHelper
{

enum class NeighborDir
{
    X, Y, Z, Count
};

// point between two neighbor voxels
struct SeparationPoint
{
    Vector3f position; // coordinate
    VertId vid; // any valid VertId is ok
    // each SeparationPointMap element has three SeparationPoint, it is not guaranteed that all three are valid (at least one is)
    // so there are some points present in map that are not valid
    explicit operator bool() const
    {
        return vid.valid();
    }
};

using SeparationPointSet = std::array<SeparationPoint, size_t( NeighborDir::Count )>;
using SeparationPointMap = ParallelHashMap<size_t, SeparationPointSet>;

// lookup table from
// http://paulbourke.net/geometry/polygonise/
using EdgeDirIndex = std::pair<int, NeighborDir>;
constexpr std::array<EdgeDirIndex, 12> cEdgeIndicesMap = {
   EdgeDirIndex{0,NeighborDir::X},
   EdgeDirIndex{1,NeighborDir::Y},
   EdgeDirIndex{2,NeighborDir::X},
   EdgeDirIndex{0,NeighborDir::Y},

   EdgeDirIndex{4,NeighborDir::X},
   EdgeDirIndex{5,NeighborDir::Y},
   EdgeDirIndex{6,NeighborDir::X},
   EdgeDirIndex{4,NeighborDir::Y},

   EdgeDirIndex{0,NeighborDir::Z},
   EdgeDirIndex{1,NeighborDir::Z},
   EdgeDirIndex{3,NeighborDir::Z},
   EdgeDirIndex{2,NeighborDir::Z}
};

const std::array<Vector3i, 8> cVoxelNeighbors{
    Vector3i{0,0,0},
    Vector3i{1,0,0},
    Vector3i{0,1,0},
    Vector3i{1,1,0},
    Vector3i{0,0,1},
    Vector3i{1,0,1},
    Vector3i{0,1,1},
    Vector3i{1,1,1}
};
const std::array<int, 8> cMapNeighborsShift{ 0,1,3,2,4,5,7,6 };

using TriangulationPlan = std::vector<int>;
const std::array<TriangulationPlan, 256> cTriangleTable = {
TriangulationPlan{},
TriangulationPlan{0, 8, 3},
TriangulationPlan{0, 1, 9},
TriangulationPlan{1, 8, 3, 9, 8, 1},
TriangulationPlan{1, 2, 10},
TriangulationPlan{0, 8, 3, 1, 2, 10},
TriangulationPlan{9, 2, 10, 0, 2, 9},
TriangulationPlan{2, 8, 3, 2, 10, 8, 10, 9, 8},
TriangulationPlan{3, 11, 2},
TriangulationPlan{0, 11, 2, 8, 11, 0},
TriangulationPlan{1, 9, 0, 2, 3, 11},
TriangulationPlan{1, 11, 2, 1, 9, 11, 9, 8, 11},
TriangulationPlan{3, 10, 1, 11, 10, 3},
TriangulationPlan{0, 10, 1, 0, 8, 10, 8, 11, 10},
TriangulationPlan{3, 9, 0, 3, 11, 9, 11, 10, 9},
TriangulationPlan{9, 8, 10, 10, 8, 11},
TriangulationPlan{4, 7, 8},
TriangulationPlan{4, 3, 0, 7, 3, 4},
TriangulationPlan{0, 1, 9, 8, 4, 7},
TriangulationPlan{4, 1, 9, 4, 7, 1, 7, 3, 1},
TriangulationPlan{1, 2, 10, 8, 4, 7},
TriangulationPlan{3, 4, 7, 3, 0, 4, 1, 2, 10},
TriangulationPlan{9, 2, 10, 9, 0, 2, 8, 4, 7},
TriangulationPlan{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4},
TriangulationPlan{8, 4, 7, 3, 11, 2},
TriangulationPlan{11, 4, 7, 11, 2, 4, 2, 0, 4},
TriangulationPlan{9, 0, 1, 8, 4, 7, 2, 3, 11},
TriangulationPlan{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1},
TriangulationPlan{3, 10, 1, 3, 11, 10, 7, 8, 4},
TriangulationPlan{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4},
TriangulationPlan{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3},
TriangulationPlan{4, 7, 11, 4, 11, 9, 9, 11, 10},
TriangulationPlan{9, 5, 4},
TriangulationPlan{9, 5, 4, 0, 8, 3},
TriangulationPlan{0, 5, 4, 1, 5, 0},
TriangulationPlan{8, 5, 4, 8, 3, 5, 3, 1, 5},
TriangulationPlan{1, 2, 10, 9, 5, 4},
TriangulationPlan{3, 0, 8, 1, 2, 10, 4, 9, 5},
TriangulationPlan{5, 2, 10, 5, 4, 2, 4, 0, 2},
TriangulationPlan{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8},
TriangulationPlan{9, 5, 4, 2, 3, 11},
TriangulationPlan{0, 11, 2, 0, 8, 11, 4, 9, 5},
TriangulationPlan{0, 5, 4, 0, 1, 5, 2, 3, 11},
TriangulationPlan{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5},
TriangulationPlan{10, 3, 11, 10, 1, 3, 9, 5, 4},
TriangulationPlan{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10},
TriangulationPlan{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3},
TriangulationPlan{5, 4, 8, 5, 8, 10, 10, 8, 11},
TriangulationPlan{9, 7, 8, 5, 7, 9},
TriangulationPlan{9, 3, 0, 9, 5, 3, 5, 7, 3},
TriangulationPlan{0, 7, 8, 0, 1, 7, 1, 5, 7},
TriangulationPlan{1, 5, 3, 3, 5, 7},
TriangulationPlan{9, 7, 8, 9, 5, 7, 10, 1, 2},
TriangulationPlan{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3},
TriangulationPlan{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2},
TriangulationPlan{2, 10, 5, 2, 5, 3, 3, 5, 7},
TriangulationPlan{7, 9, 5, 7, 8, 9, 3, 11, 2},
TriangulationPlan{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11},
TriangulationPlan{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7},
TriangulationPlan{11, 2, 1, 11, 1, 7, 7, 1, 5},
TriangulationPlan{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11},
TriangulationPlan{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0},
TriangulationPlan{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0},
TriangulationPlan{11, 10, 5, 7, 11, 5},
TriangulationPlan{10, 6, 5},
TriangulationPlan{0, 8, 3, 5, 10, 6},
TriangulationPlan{9, 0, 1, 5, 10, 6},
TriangulationPlan{1, 8, 3, 1, 9, 8, 5, 10, 6},
TriangulationPlan{1, 6, 5, 2, 6, 1},
TriangulationPlan{1, 6, 5, 1, 2, 6, 3, 0, 8},
TriangulationPlan{9, 6, 5, 9, 0, 6, 0, 2, 6},
TriangulationPlan{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8},
TriangulationPlan{2, 3, 11, 10, 6, 5},
TriangulationPlan{11, 0, 8, 11, 2, 0, 10, 6, 5},
TriangulationPlan{0, 1, 9, 2, 3, 11, 5, 10, 6},
TriangulationPlan{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11},
TriangulationPlan{6, 3, 11, 6, 5, 3, 5, 1, 3},
TriangulationPlan{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6},
TriangulationPlan{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9},
TriangulationPlan{6, 5, 9, 6, 9, 11, 11, 9, 8},
TriangulationPlan{5, 10, 6, 4, 7, 8},
TriangulationPlan{4, 3, 0, 4, 7, 3, 6, 5, 10},
TriangulationPlan{1, 9, 0, 5, 10, 6, 8, 4, 7},
TriangulationPlan{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4},
TriangulationPlan{6, 1, 2, 6, 5, 1, 4, 7, 8},
TriangulationPlan{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7},
TriangulationPlan{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6},
TriangulationPlan{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9},
TriangulationPlan{3, 11, 2, 7, 8, 4, 10, 6, 5},
TriangulationPlan{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11},
TriangulationPlan{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6},
TriangulationPlan{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6},
TriangulationPlan{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6},
TriangulationPlan{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11},
TriangulationPlan{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7},
TriangulationPlan{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9},
TriangulationPlan{10, 4, 9, 6, 4, 10},
TriangulationPlan{4, 10, 6, 4, 9, 10, 0, 8, 3},
TriangulationPlan{10, 0, 1, 10, 6, 0, 6, 4, 0},
TriangulationPlan{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10},
TriangulationPlan{1, 4, 9, 1, 2, 4, 2, 6, 4},
TriangulationPlan{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4},
TriangulationPlan{0, 2, 4, 4, 2, 6},
TriangulationPlan{8, 3, 2, 8, 2, 4, 4, 2, 6},
TriangulationPlan{10, 4, 9, 10, 6, 4, 11, 2, 3},
TriangulationPlan{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6},
TriangulationPlan{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10},
TriangulationPlan{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1},
TriangulationPlan{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3},
TriangulationPlan{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1},
TriangulationPlan{3, 11, 6, 3, 6, 0, 0, 6, 4},
TriangulationPlan{6, 4, 8, 11, 6, 8},
TriangulationPlan{7, 10, 6, 7, 8, 10, 8, 9, 10},
TriangulationPlan{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10},
TriangulationPlan{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0},
TriangulationPlan{10, 6, 7, 10, 7, 1, 1, 7, 3},
TriangulationPlan{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7},
TriangulationPlan{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9},
TriangulationPlan{7, 8, 0, 7, 0, 6, 6, 0, 2},
TriangulationPlan{7, 3, 2, 6, 7, 2},
TriangulationPlan{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7},
TriangulationPlan{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7},
TriangulationPlan{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11},
TriangulationPlan{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1},
TriangulationPlan{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6},
TriangulationPlan{0, 9, 1, 11, 6, 7},
TriangulationPlan{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0},
TriangulationPlan{7, 11, 6},
TriangulationPlan{7, 6, 11},
TriangulationPlan{3, 0, 8, 11, 7, 6},
TriangulationPlan{0, 1, 9, 11, 7, 6},
TriangulationPlan{8, 1, 9, 8, 3, 1, 11, 7, 6},
TriangulationPlan{10, 1, 2, 6, 11, 7},
TriangulationPlan{1, 2, 10, 3, 0, 8, 6, 11, 7},
TriangulationPlan{2, 9, 0, 2, 10, 9, 6, 11, 7},
TriangulationPlan{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8},
TriangulationPlan{7, 2, 3, 6, 2, 7},
TriangulationPlan{7, 0, 8, 7, 6, 0, 6, 2, 0},
TriangulationPlan{2, 7, 6, 2, 3, 7, 0, 1, 9},
TriangulationPlan{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6},
TriangulationPlan{10, 7, 6, 10, 1, 7, 1, 3, 7},
TriangulationPlan{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8},
TriangulationPlan{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7},
TriangulationPlan{7, 6, 10, 7, 10, 8, 8, 10, 9},
TriangulationPlan{6, 8, 4, 11, 8, 6},
TriangulationPlan{3, 6, 11, 3, 0, 6, 0, 4, 6},
TriangulationPlan{8, 6, 11, 8, 4, 6, 9, 0, 1},
TriangulationPlan{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6},
TriangulationPlan{6, 8, 4, 6, 11, 8, 2, 10, 1},
TriangulationPlan{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6},
TriangulationPlan{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9},
TriangulationPlan{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3},
TriangulationPlan{8, 2, 3, 8, 4, 2, 4, 6, 2},
TriangulationPlan{0, 4, 2, 4, 6, 2},
TriangulationPlan{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8},
TriangulationPlan{1, 9, 4, 1, 4, 2, 2, 4, 6},
TriangulationPlan{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1},
TriangulationPlan{10, 1, 0, 10, 0, 6, 6, 0, 4},
TriangulationPlan{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3},
TriangulationPlan{10, 9, 4, 6, 10, 4},
TriangulationPlan{4, 9, 5, 7, 6, 11},
TriangulationPlan{0, 8, 3, 4, 9, 5, 11, 7, 6},
TriangulationPlan{5, 0, 1, 5, 4, 0, 7, 6, 11},
TriangulationPlan{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5},
TriangulationPlan{9, 5, 4, 10, 1, 2, 7, 6, 11},
TriangulationPlan{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5},
TriangulationPlan{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2},
TriangulationPlan{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6},
TriangulationPlan{7, 2, 3, 7, 6, 2, 5, 4, 9},
TriangulationPlan{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7},
TriangulationPlan{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0},
TriangulationPlan{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8},
TriangulationPlan{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7},
TriangulationPlan{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4},
TriangulationPlan{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10},
TriangulationPlan{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10},
TriangulationPlan{6, 9, 5, 6, 11, 9, 11, 8, 9},
TriangulationPlan{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5},
TriangulationPlan{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11},
TriangulationPlan{6, 11, 3, 6, 3, 5, 5, 3, 1},
TriangulationPlan{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6},
TriangulationPlan{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10},
TriangulationPlan{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5},
TriangulationPlan{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3},
TriangulationPlan{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2},
TriangulationPlan{9, 5, 6, 9, 6, 0, 0, 6, 2},
TriangulationPlan{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8},
TriangulationPlan{1, 5, 6, 2, 1, 6},
TriangulationPlan{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6},
TriangulationPlan{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0},
TriangulationPlan{0, 3, 8, 5, 6, 10},
TriangulationPlan{10, 5, 6},
TriangulationPlan{11, 5, 10, 7, 5, 11},
TriangulationPlan{11, 5, 10, 11, 7, 5, 8, 3, 0},
TriangulationPlan{5, 11, 7, 5, 10, 11, 1, 9, 0},
TriangulationPlan{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1},
TriangulationPlan{11, 1, 2, 11, 7, 1, 7, 5, 1},
TriangulationPlan{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11},
TriangulationPlan{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7},
TriangulationPlan{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2},
TriangulationPlan{2, 5, 10, 2, 3, 5, 3, 7, 5},
TriangulationPlan{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5},
TriangulationPlan{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2},
TriangulationPlan{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2},
TriangulationPlan{1, 3, 5, 3, 7, 5},
TriangulationPlan{0, 8, 7, 0, 7, 1, 1, 7, 5},
TriangulationPlan{9, 0, 3, 9, 3, 5, 5, 3, 7},
TriangulationPlan{9, 8, 7, 5, 9, 7},
TriangulationPlan{5, 8, 4, 5, 10, 8, 10, 11, 8},
TriangulationPlan{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0},
TriangulationPlan{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5},
TriangulationPlan{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4},
TriangulationPlan{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8},
TriangulationPlan{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11},
TriangulationPlan{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5},
TriangulationPlan{9, 4, 5, 2, 11, 3},
TriangulationPlan{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4},
TriangulationPlan{5, 10, 2, 5, 2, 4, 4, 2, 0},
TriangulationPlan{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9},
TriangulationPlan{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2},
TriangulationPlan{8, 4, 5, 8, 5, 3, 3, 5, 1},
TriangulationPlan{0, 4, 5, 1, 0, 5},
TriangulationPlan{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5},
TriangulationPlan{9, 4, 5},
TriangulationPlan{4, 11, 7, 4, 9, 11, 9, 10, 11},
TriangulationPlan{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11},
TriangulationPlan{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11},
TriangulationPlan{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4},
TriangulationPlan{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2},
TriangulationPlan{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3},
TriangulationPlan{11, 7, 4, 11, 4, 2, 2, 4, 0},
TriangulationPlan{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4},
TriangulationPlan{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9},
TriangulationPlan{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7},
TriangulationPlan{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10},
TriangulationPlan{1, 10, 2, 8, 7, 4},
TriangulationPlan{4, 9, 1, 4, 1, 7, 7, 1, 3},
TriangulationPlan{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1},
TriangulationPlan{4, 0, 3, 7, 4, 3},
TriangulationPlan{4, 8, 7},
TriangulationPlan{9, 10, 8, 10, 11, 8},
TriangulationPlan{3, 0, 9, 3, 9, 11, 11, 9, 10},
TriangulationPlan{0, 1, 10, 0, 10, 8, 8, 10, 11},
TriangulationPlan{3, 1, 10, 11, 3, 10},
TriangulationPlan{1, 2, 11, 1, 11, 9, 9, 11, 8},
TriangulationPlan{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9},
TriangulationPlan{0, 2, 11, 8, 0, 11},
TriangulationPlan{3, 2, 11},
TriangulationPlan{2, 3, 8, 2, 8, 10, 10, 8, 9},
TriangulationPlan{9, 10, 2, 0, 9, 2},
TriangulationPlan{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8},
TriangulationPlan{1, 10, 2},
TriangulationPlan{1, 3, 8, 9, 1, 8},
TriangulationPlan{0, 9, 1},
TriangulationPlan{0, 3, 8},
TriangulationPlan{}
};

}

using namespace MarchingCubesHelper;
using ConstAccessor = openvdb::FloatGrid::ConstAccessor;
using VdbCoord = openvdb::Coord;

Vector3f voxelPositionerLinear( const Vector3f& pos0, const Vector3f& pos1, float v0, float v1, float iso )
{
    auto ratio = std::clamp( std::abs( iso - v0 ) / std::abs( v1 - v0 ), 0.0f, 1.0f );
    return ( 1.0f - ratio ) * pos0 + ratio * pos1;
}

SeparationPoint findSeparationPoint( const VdbVolume& volume, const ConstAccessor& acc, const VdbCoord& minCoord,
    const VolumeIndexer& indexer, VoxelId base, NeighborDir dir, const VolumeToMeshParams& params )
{
    auto basePos = indexer.toPos( base );
    auto shift = ( 1 << params.neighborVoxExp );
    if ( basePos[int( dir )] + shift >= volume.dims[int( dir )] )
        return {};
    auto coord = openvdb::Coord{ basePos.x + minCoord.x(),basePos.y + minCoord.y(),basePos.z + minCoord.z() };
    const auto& valueB = acc.getValue( coord );// volume.data[base];
    auto nextCoord = coord;
    nextCoord[int( dir )] += shift;
    const auto& valueD = acc.getValue( nextCoord );// volume.data[nextId];
    bool bLower = valueB < params.iso;
    bool dLower = valueD < params.iso;

    if ( bLower == dLower )
        return {};

    SeparationPoint res;
    Vector3f coordF = Vector3f( float( coord.x() ), float( coord.y() ), float( coord.z() ) );
    Vector3f nextCoordF = Vector3f( float( nextCoord.x() ), float( nextCoord.y() ), float( nextCoord.z() ) );
    auto bPos = params.basis.b + params.basis.A * coordF;
    auto dPos = params.basis.b + params.basis.A * nextCoordF;
    res.position = params.positioner( bPos, dPos, valueB, valueD, params.iso );
    res.vid = VertId{ 0 };
    return res;
}

SeparationPoint findSeparationPoint( const SimpleVolume& volume,
    const VolumeIndexer& indexer, VoxelId base, NeighborDir dir, const VolumeToMeshParams& params )
{
    auto basePos = indexer.toPos( base );
    auto nextPos = basePos;
    nextPos[int( dir )] += ( 1 << params.neighborVoxExp );
    if ( nextPos[int( dir )] >= volume.dims[int( dir )] )
        return {};
    const auto& valueB = volume.data[base];
    const auto& valueD = volume.data[indexer.toVoxelId( nextPos ).get()];
    if ( std::isnan( valueB ) || std::isnan( valueD ) )
        return {};
    bool bLower = valueB < params.iso;
    bool dLower = valueD < params.iso;

    if ( bLower == dLower )
        return {};

    SeparationPoint res;
    Vector3f coordF = Vector3f( basePos ) + Vector3f::diagonal( 0.5f );
    Vector3f nextCoordF = Vector3f( nextPos ) + Vector3f::diagonal( 0.5f );
    auto bPos = params.basis.b + params.basis.A * coordF;
    auto dPos = params.basis.b + params.basis.A * nextCoordF;
    res.position = params.positioner( bPos, dPos, valueB, valueD, params.iso );
    res.vid = VertId{ 0 };
    return res;
}

template<typename V>
std::optional<Mesh> volumeToMesh( const V& volume, const VolumeToMeshParams& params /*= {} */ )
{
    if constexpr ( std::is_same_v<V, VdbVolume> )
        if ( !volume.data )
            return {};

    if ( params.iso <= volume.min || params.iso >= volume.max ||
        volume.dims.x <= 0 || volume.dims.y <= 0 || volume.dims.z <= 0 )
        return Mesh();

    MR_TIMER

    VdbCoord minCoord;
    if constexpr ( std::is_same_v<V, VdbVolume> )
        minCoord = volume.data->evalActiveVoxelBoundingBox().min();

    VolumeIndexer indexer( volume.dims );

    std::atomic<bool> keepGoing{ true };
    auto mainThreadId = std::this_thread::get_id();
    int lastSubMap = -1;

    SeparationPointMap hmap;
    const auto subcnt = hmap.subcnt();
    auto voxShift = ( 1 << params.neighborVoxExp );
    // find all separate points
    // fill map in parallel
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subcnt, 1 ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        std::optional<ConstAccessor> acc;
        if constexpr ( std::is_same_v<V, VdbVolume> )
            acc = volume.data->getConstAccessor();
        for ( size_t i = 0; i < indexer.size(); ++i )
        {
            if ( params.cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            auto pos = indexer.toPos( VoxelId( i ) );
            if ( ( pos.x % voxShift ) != 0 || 
                ( pos.y % voxShift ) != 0 ||
                ( pos.z % voxShift ) != 0 )
                continue;

            auto hashval = hmap.hash( i );
            if ( hmap.subidx( hashval ) != range.begin() )
                continue;

            SeparationPointSet set;
            bool atLeastOneOk = false;
            for ( int n = int( NeighborDir::X ); n < int( NeighborDir::Count ); ++n )
            {
                SeparationPoint separation;
                if constexpr ( std::is_same_v<V, VdbVolume> )
                    separation = findSeparationPoint( volume, *acc, minCoord, indexer, VoxelId( i ), NeighborDir( n ), params );
                else
                    separation = findSeparationPoint( volume, indexer, VoxelId( i ), NeighborDir( n ), params );

                if ( separation )
                {
                    set[n] = std::move( separation );
                    atLeastOneOk = true;
                }
            }

            if ( params.cb && ( ( i % 1024 ) == 0 ) && std::this_thread::get_id() == mainThreadId &&
                ( lastSubMap == -1 || lastSubMap == range.begin() ) )
            {
                if ( lastSubMap == -1 )
                    lastSubMap = int( range.begin() );
                if ( !params.cb( 0.3f * float( i ) / float( indexer.size() ) ) )
                    keepGoing.store( false, std::memory_order_relaxed );
            }

            if ( !atLeastOneOk )
                continue;

            hmap.insert( { i, set } );
        }
    } );

    if ( params.cb && !keepGoing )
        return {};

    // numerate verts in parallel (to have packed mesh as result, determined numeration independent of thread number)
    struct VertsNumeration
    {
        // explicit ctor to fix clang build with `vec.emplace_back( ind, 0 )`
        VertsNumeration( size_t ind, size_t num ) :initIndex{ ind }, numVerts{ num }{}
        size_t initIndex{ 0 };
        size_t numVerts{ 0 };
    };
    using PerThreadVertNumeration = std::vector<VertsNumeration>;
    tbb::enumerable_thread_specific<PerThreadVertNumeration> perThreadVertNumeration;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, indexer.size() ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        auto& localNumeration = perThreadVertNumeration.local();
        localNumeration.emplace_back( range.begin(), 0 );
        auto& thisRangeNumeration = localNumeration.back().numVerts;
        for ( size_t ind = range.begin(); ind < range.end(); ++ind )
        {
            // as far as map is filled, change of each individual value should be safe
            auto mapIter = hmap.find( ind );
            if ( mapIter == hmap.cend() )
                continue;
            for ( auto& dir : mapIter->second )
                if ( dir )
                    dir.vid = VertId( thisRangeNumeration++ );
        }
    }, tbb::static_partitioner() );// static_partitioner to have bigger grains

    // organize vert numeration
    std::vector<VertsNumeration> resultVertNumeration;
    for ( auto& perThreadNum : perThreadVertNumeration )
    {
        // remove empty
        perThreadNum.erase( std::remove_if( perThreadNum.begin(), perThreadNum.end(),
            [] ( const auto& obj )
        {
            return obj.numVerts == 0;
        } ), perThreadNum.end() );
        if ( perThreadNum.empty() )
            continue;
        // accum not empty
        resultVertNumeration.insert( resultVertNumeration.end(),
            std::make_move_iterator( perThreadNum.begin() ), std::make_move_iterator( perThreadNum.end() ) );
    }
    // sort by voxel index
    std::sort( resultVertNumeration.begin(), resultVertNumeration.end(), [] ( const auto& l, const auto& r )
    {
        return l.initIndex < r.initIndex;
    } );

    auto getVertIndexShiftForVoxelId = [&] ( size_t ind )
    {
        size_t shift = 0;
        for ( int i = 1; i < resultVertNumeration.size(); ++i )
        {
            if ( ind >= resultVertNumeration[i].initIndex )
                shift += resultVertNumeration[i - 1].numVerts;
        }
        return VertId( shift );
    };

    // update map with determined vert indices
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subcnt, 1 ),
    [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        hmap.with_submap( range.begin(), [&] ( const SeparationPointMap::EmbeddedSet& subSet )
        {
            // const_cast here is safe, we don't write to map, just change internal data
            for ( auto& [ind, set] : const_cast< SeparationPointMap::EmbeddedSet& >( subSet ) )
            {
                auto vertShift = getVertIndexShiftForVoxelId( ind );
                for ( auto& sepPoint : set )
                    if ( sepPoint )
                        sepPoint.vid += vertShift;
            }
        } );
    } );


    if ( params.cb && !params.cb( 0.5f ) )
        return {};

    // check neighbor iterator valid
    auto checkIter = [&] ( const auto& iter, int mode ) -> bool
    {
        switch ( mode )
        {
        case 0: // base voxel
            return iter != hmap.cend();
        case 1: // x + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return iter->second[int( NeighborDir::Y )] || iter->second[int( NeighborDir::Z )];
        }
        case 2: // y + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return iter->second[int( NeighborDir::X )] || iter->second[int( NeighborDir::Z )];
        }
        case 3: // x + 1, y + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return bool( iter->second[int( NeighborDir::Z )] );
        }
        case 4: // z + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return iter->second[int( NeighborDir::X )] || iter->second[int( NeighborDir::Y )];
        }
        case 5: // x + 1, z + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return bool( iter->second[int( NeighborDir::Y )] );
        }
        case 6: // y + 1, z + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return bool( iter->second[int( NeighborDir::X )] );
        }
        default:
            return false;
        }
    };

    // triangulate by table
    struct TriangulationData
    {
        size_t initInd{ 0 }; // this is needed to have determined topology independent of threads number
        Triangulation t;
        Vector<VoxelId, FaceId> faceMap;
    };
    using PerThreadTriangulation = std::vector<TriangulationData>;
    tbb::enumerable_thread_specific<PerThreadTriangulation> triangulationPerThread;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, indexer.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        // setup local triangulation
        auto& localTriangulatoinData = triangulationPerThread.local();
        localTriangulatoinData.emplace_back();
        auto& thisTriData = localTriangulatoinData.back();
        thisTriData.initInd = range.begin();
        auto& t = thisTriData.t;
        auto& faceMap = thisTriData.faceMap;

        // cell data
        std::optional<ConstAccessor> acc;
        if constexpr ( std::is_same_v<V, VdbVolume> )
            acc = volume.data->getConstAccessor();
        std::array<SeparationPointMap::const_iterator, 7> iters;
        std::array<bool, 7> iterStatus;
        unsigned char voxelConfiguration;
        for ( size_t ind = range.begin(); ind < range.end(); ++ind )
        {
            if ( params.cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;
            Vector3i basePos = indexer.toPos( VoxelId( ind ) );
            if ( basePos.x + voxShift >= volume.dims.x ||
                basePos.y + voxShift >= volume.dims.y ||
                basePos.z + voxShift >= volume.dims.z )
                continue;

            if ( ( basePos.x % voxShift ) != 0 ||
                ( basePos.y % voxShift ) != 0 ||
                ( basePos.z % voxShift ) != 0 )
                continue;

            bool voxelValid = false;
            for ( int i = 0; i < iters.size(); ++i )
            {
                iters[i] = hmap.find( size_t( indexer.toVoxelId( basePos + voxShift * cVoxelNeighbors[i] ) ) );
                iterStatus[i] = checkIter( iters[i], i );
                if ( !voxelValid && iterStatus[i] )
                    voxelValid = true;
            }

            if ( params.cb && ( ( ind % 1024 ) == 0 ) && std::this_thread::get_id() == mainThreadId )
            {
                if ( !params.cb( 0.5f + 0.35f * float( ind ) / float( range.size() ) ) )
                    keepGoing.store( false, std::memory_order_relaxed );
            }

            if ( !voxelValid )
                continue;
            voxelConfiguration = 0;
            [[maybe_unused]] bool atLeastOneNan = false;
            for ( int i = 0; i < cVoxelNeighbors.size(); ++i )
            {
                auto pos = basePos + voxShift * cVoxelNeighbors[i];
                float value{ 0.0f };
                if constexpr ( std::is_same_v<V, VdbVolume> )
                    value = acc->getValue( { pos.x + minCoord.x(),pos.y + minCoord.y(),pos.z + minCoord.z() } );
                else
                {
                    value = volume.data[indexer.toVoxelId( pos ).get()];
                    // find non nan neighbor
                    constexpr std::array<uint8_t, 7> cNeighborsOrder{
                        0b001,
                        0b010,
                        0b100,
                        0b011,
                        0b101,
                        0b110,
                        0b111
                    };
                    int neighIndex = 0;
                    // iterates over nan neighbors to find consistent value
                    while ( std::isnan( value ) && neighIndex < 7 )
                    {
                        auto neighPos = pos;
                        for ( int posCoord = 0; posCoord < 3; ++posCoord )
                        {
                            int sign = 1;
                            if ( cVoxelNeighbors[i][posCoord] == 1 )
                                sign = -1;
                            neighPos[posCoord] += ( sign * voxShift *
                                ( ( cNeighborsOrder[neighIndex] & ( 1 << posCoord ) ) >> posCoord ) );
                        }
                        value = volume.data[indexer.toVoxelId( neighPos ).get()];
                        ++neighIndex;
                    }
                    if ( std::isnan( value ) )
                    {
                        voxelValid = false;
                        break;
                    }
                    if ( !atLeastOneNan && neighIndex > 0 )
                        atLeastOneNan = true;
                }
                if ( value >= params.iso )
                    continue;
                voxelConfiguration |= ( 1 << cMapNeighborsShift[i] );
            }

            if constexpr ( std::is_same_v<V, SimpleVolume> )
            {
                // ensure consistent nan voxel
                if ( atLeastOneNan && voxelValid )
                {
                    const auto& plan = cTriangleTable[voxelConfiguration];
                    for ( int i = 0; i < plan.size() && voxelValid; i += 3 )
                    {
                        const auto& [interIndex0, dir0] = cEdgeIndicesMap[plan[i]];
                        const auto& [interIndex1, dir1] = cEdgeIndicesMap[plan[i + 1]];
                        const auto& [interIndex2, dir2] = cEdgeIndicesMap[plan[i + 2]];
                        // `iterStatus` indicates that current voxel has valid point for desired triangulation
                        // as far as iter has 3 directions we use `dir` to validate (make sure that there is point in needed edge) desired direction
                        voxelValid = voxelValid && ( iterStatus[interIndex0] && iters[interIndex0]->second[int( dir0 )].vid );
                        voxelValid = voxelValid && ( iterStatus[interIndex1] && iters[interIndex1]->second[int( dir1 )].vid );
                        voxelValid = voxelValid && ( iterStatus[interIndex2] && iters[interIndex2]->second[int( dir2 )].vid );
                    }
                }
            }

            if constexpr ( std::is_same_v<V, SimpleVolume> )
                if ( !voxelValid )
                    continue;

            const auto& plan = cTriangleTable[voxelConfiguration];
            for ( int i = 0; i < plan.size(); i += 3 )
            {
                const auto& [interIndex0, dir0] = cEdgeIndicesMap[plan[i]];
                const auto& [interIndex1, dir1] = cEdgeIndicesMap[plan[i + 1]];
                const auto& [interIndex2, dir2] = cEdgeIndicesMap[plan[i + 2]];
                assert( iterStatus[interIndex0] && iters[interIndex0]->second[int( dir0 )].vid );
                assert( iterStatus[interIndex1] && iters[interIndex1]->second[int( dir1 )].vid );
                assert( iterStatus[interIndex2] && iters[interIndex2]->second[int( dir2 )].vid );

                if ( params.lessInside )
                    t.emplace_back( ThreeVertIds{
                    iters[interIndex0]->second[int( dir0 )].vid,
                    iters[interIndex2]->second[int( dir2 )].vid,
                    iters[interIndex1]->second[int( dir1 )].vid
                    } );
                else
                    t.emplace_back( ThreeVertIds{
                    iters[interIndex0]->second[int( dir0 )].vid,
                    iters[interIndex1]->second[int( dir1 )].vid,
                    iters[interIndex2]->second[int( dir2 )].vid
                    } );
                if ( params.outVoxelPerFaceMap )
                    faceMap.emplace_back( VoxelId{ ind } );
            }
        }
    }, tbb::static_partitioner() );

    if ( params.cb && !keepGoing )
        return {};

    // organize per thread triangulation
    std::vector<TriangulationData> resTriangulatoinData;
    for ( auto& threadTriData : triangulationPerThread )
    {
        // remove empty
        threadTriData.erase( std::remove_if( threadTriData.begin(), threadTriData.end(),
            [] ( const auto& obj )
        {
            return obj.t.empty();
        } ), threadTriData.end() );
        if ( threadTriData.empty() )
            continue;
        // accum not empty
        resTriangulatoinData.insert( resTriangulatoinData.end(),
            std::make_move_iterator( threadTriData.begin() ), std::make_move_iterator( threadTriData.end() ) );
    }
    // sort by voxel index
    std::sort( resTriangulatoinData.begin(), resTriangulatoinData.end(), [] ( const auto& l, const auto& r )
    {
        return l.initInd < r.initInd;
    } );

    // create result triangulation
    Mesh result;
    Triangulation resTriangulation;
    if ( params.outVoxelPerFaceMap )
        params.outVoxelPerFaceMap->clear();
    for ( auto& [ind, t, faceMap] : resTriangulatoinData )
    {
        resTriangulation.vec_.insert( resTriangulation.vec_.end(),
            std::make_move_iterator( t.vec_.begin() ), std::make_move_iterator( t.vec_.end() ) );
        if ( params.outVoxelPerFaceMap )
            params.outVoxelPerFaceMap->vec_.insert( params.outVoxelPerFaceMap->vec_.end(),
                std::make_move_iterator( faceMap.vec_.begin() ), std::make_move_iterator( faceMap.vec_.end() ) );
    }
    result.topology = MeshBuilder::fromTriangles( std::move( resTriangulation ) );
    result.points.resize( result.topology.lastValidVert() + 1 );

    if ( params.cb && !params.cb( 0.95f ) )
        return {};

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subcnt, 1 ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        hmap.with_submap( range.begin(), [&] ( const SeparationPointMap::EmbeddedSet& subSet )
        {
            for ( auto& separation : subSet )
            {
                for ( int i = int( NeighborDir::X ); i < int( NeighborDir::Count ); ++i )
                    if ( separation.second[i].vid.valid() )
                        result.points[separation.second[i].vid] = separation.second[i].position;
            }
        } );
    } );

    if ( params.cb && !params.cb( 1.0f ) )
        return {};

    return result;
}

std::optional<Mesh> simpleVolumeToMesh( const SimpleVolume& volume, const VolumeToMeshParams& params /*= {} */ )
{
    return volumeToMesh( volume, params );
}
std::optional<Mesh> vdbVolumeToMesh( const VdbVolume& volume, const VolumeToMeshParams& params /*= {} */ )
{
    return volumeToMesh( volume, params );
}

}
#endif