#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRMeshToDistanceVolume.h"
#include "MRIsNaN.h"
#include "MRMesh.h"
#include "MRSimpleVolume.h"
#include "MRTimer.h"
#include "MRVolumeIndexer.h"
#include "MRIntersectionPrecomputes.h"
#include "MRFastWindingNumber.h"
#include "MRLine3.h"
#include "MRMeshIntersect.h"
#include "MRPch/MRTBB.h"
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

Expected<SimpleVolume, std::string> meshToDistanceVolume( const Mesh& mesh, const MeshToDistanceVolumeParams& params /*= {} */ )
{
    MR_TIMER
    SimpleVolume res;
    res.voxelSize = params.voxelSize;
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

        auto basis = AffineXf3f::linear( Matrix3f::scale( params.voxelSize ) );
        basis.b = params.origin;
        constexpr float beta = 2;
        if ( auto d = fwn->calcFromGridWithDistances( res.data, res.dims, Vector3f::diagonal( 0.5f ), Vector3f::diagonal( 1.0f ), basis, beta,
            params.maxDistSq, params.minDistSq, params.cb ); !d )
        {
            return unexpected( std::move( d.error() ) );
        }
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
            auto voxelCenter = params.origin + mult( params.voxelSize, coord );
            float dist{ 0.0f };
            if ( params.signMode != SignDetectionMode::ProjectionNormal )
                dist = std::sqrt( findProjection( voxelCenter, mesh, params.maxDistSq, nullptr, params.minDistSq ).distSq );
            else
            {
                auto s = findSignedDistance( voxelCenter, mesh, params.maxDistSq, params.minDistSq );
                dist = s ? s->dist : cQuietNan;
            }

            if ( !isNanFast( dist ) )
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
        return unexpectedOperationCanceled();
    for ( const auto& [min, max] : minMax )
    {
        if ( min < res.min )
            res.min = min;
        if ( max > res.max )
            res.max = max;
    }
    return res;
}

Expected<SimpleVolume, std::string> meshToSimpleVolume( const Mesh& mesh, const MeshToDistanceVolumeParams& params )
{
    return meshToDistanceVolume( mesh, params );
}

} //namespace MR
#endif
