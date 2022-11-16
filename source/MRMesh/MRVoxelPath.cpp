#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRVoxelPath.h"
#include "MRFloatGrid.h"
#include "MRSimpleVolume.h"
#include "MRVector3.h"
#include "MRTimer.h"
#include <cfloat>
#include <parallel_hashmap/phmap.h>
#include <queue>
#include <cmath>
#include <filesystem>

namespace MR
{

constexpr size_t InvalidVoxel = ~size_t( 0 );

openvdb::math::Coord getCoord( size_t dimsXY, int dimsX, size_t v )
{
    int z = int( v / dimsXY );
    int sumZ = int( v % dimsXY );
    int y = sumZ / dimsX;
    int x = sumZ % dimsX;
    return {x,y,z};
}

float getDistSq( size_t dimsXY, int dimsX, size_t a, size_t b )
{
    auto coordA = getCoord( dimsXY, dimsX, a );
    auto coordB = getCoord( dimsXY, dimsX, b );
    Vector3i difs( (int) coordA.x() - coordB.x(), (int) coordA.y() - coordB.y(), (int) coordA.z() - coordB.z() );
    return float( difs.lengthSq() );
}

struct QuarterParams
{
    openvdb::Coord start;
    openvdb::Coord stop;
    openvdb::Coord diff;
    openvdb::Coord absDiff;
};

QuarterParams setupQuaterParams( size_t dimsXY, int dimsX, size_t strat, size_t stop )
{
    auto startCoord = getCoord( dimsXY, dimsX, strat );
    auto stopCoord = getCoord( dimsXY, dimsX, stop );
    auto diff = stopCoord - startCoord;
    openvdb::Coord absDiff = {std::abs( diff.x() ),std::abs( diff.y() ),std::abs( diff.z() )};
    return {startCoord,stopCoord,diff,absDiff};
}

bool isInQuater( size_t dimsXY, int dimsX, const QuarterParams& params, size_t next, char mask )
{
    if ( mask == QuarterBit::All )
        return true;

    auto coordNext = getCoord( dimsXY, dimsX, next );
    auto mainAxis = params.absDiff.maxIndex();

    auto ratio = float( coordNext[mainAxis] - params.start[mainAxis] ) / float( params.diff[mainAxis] );
    auto other1Axis = ( mainAxis + 1 ) % 3;
    auto other2Axis = ( mainAxis + 2 ) % 3;
    if ( params.absDiff[other2Axis] > params.absDiff[other1Axis] )
        std::swap( other1Axis, other2Axis );

    Vector3f cordOnAxis( params.diff.x() * ratio + params.start.x(), 
                         params.diff.y() * ratio + params.start.y(),
                         params.diff.z() * ratio + params.start.z() );

    int refOther1 = int(cordOnAxis[int(other1Axis)]);
    int refOther2 = int(cordOnAxis[int(other2Axis)]);

    char currentQuater;
    bool firstLeft{coordNext[other1Axis] < refOther1};
    bool secondLeft{coordNext[other2Axis] < refOther2};
    if ( firstLeft && secondLeft )
        currentQuater = LeftLeft;
    else if ( firstLeft )
        currentQuater = LeftRight;
    else if ( secondLeft )
        currentQuater = RightLeft;
    else
        currentQuater = RightRight;

    auto startDif = coordNext - params.start;
    auto stopDif = coordNext - params.stop;
    auto startDiffMagnitudeSq = startDif[0] * startDif[0] + startDif[1] * startDif[1] + startDif[2] * startDif[2];
    auto stopDiffMagnitudeSq = stopDif[0] * stopDif[0] + stopDif[1] * stopDif[1] + stopDif[2] * stopDif[2];

    // if point is closer then 2 voxels to start or stop, allow it to be anywhere
    if ( startDiffMagnitudeSq < 4 || stopDiffMagnitudeSq < 4 )
        return true;

    return bool( currentQuater & mask );
}

struct VoxelPathInfo
{
    // current voxel
    size_t currVoxel{InvalidVoxel};
    // predecessor voxels
    size_t prevVoxel{InvalidVoxel};
    // best summed metric to reach this vertex
    float metric = FLT_MAX;

    bool isStart() const
    {
        return prevVoxel == InvalidVoxel;
    }
};

// smaller metric to be the first
inline bool operator <( const VoxelPathInfo & a, const VoxelPathInfo & b )
{
    return a.metric > b.metric;
}

using VoxelPathInfoMap = ParallelHashMap<size_t, VoxelPathInfo>;

class VoxelsPathsBuilder
{
public:
    VoxelsPathsBuilder( const VdbVolume& voxels, const VoxelsMetric & metric );
    void addPathStart( size_t voxelFromStart, float startMetric );
    // include one more voxel in the voxels forest, returning newly reached voxel
    size_t growOneVoxel();

public:
    // returns true if further voxels forest growth is impossible
    bool done() const
    {
        return nextSteps_.empty();
    }

    // returns the path in the forest from given voxel to one of start voxles
    std::vector<size_t> getPathBack( size_t backpathStart ) const;

private:
    const VdbVolume& vdbVolume_;
    openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> accessor_;
    size_t dimXY_;
    VoxelsMetric metric_;
    VoxelPathInfoMap voxelPathInfoMap_;
    std::priority_queue<VoxelPathInfo> nextSteps_;

    // compares proposed step with the value known for current voxel;
    // if proposed step is smaller then adds it in the queue
    void addNextStep_( const VoxelPathInfo & c );
    // adds steps for all neighbors of current voxel
    void addNeigboursSteps_( float orgMetric, size_t back );
};

VoxelsPathsBuilder::VoxelsPathsBuilder( const VdbVolume& voxels, const VoxelsMetric & metric ) :
    vdbVolume_{voxels},
    accessor_{voxels.data->getConstAccessor()},
    metric_{metric}
{
    dimXY_ = size_t( voxels.dims.x ) * voxels.dims.y;
}

void VoxelsPathsBuilder::addPathStart( size_t voxelFromStart, float startMetric )
{
    MR_TIMER;
    auto & vi = voxelPathInfoMap_[voxelFromStart];
    vi.metric = std::min( vi.metric, startMetric );
    addNeigboursSteps_( vi.metric, voxelFromStart );
}

size_t VoxelsPathsBuilder::growOneVoxel()
{
    while ( !nextSteps_.empty() )
    {
        const auto c = nextSteps_.top();
        nextSteps_.pop();
        auto & vi = voxelPathInfoMap_[c.currVoxel];
        if ( vi.metric < c.metric )
        {
            // shorter path to the vertex was found
            continue;
        }
        assert( vi.metric == c.metric );
        addNeigboursSteps_( c.metric, c.currVoxel );
        return c.currVoxel;
    }
    return InvalidVoxel;
}

std::vector<size_t> VoxelsPathsBuilder::getPathBack( size_t v ) const
{
    MR_TIMER;
    std::vector<size_t> res;
    res.push_back( v );
    for ( ;;)
    {
        auto it = voxelPathInfoMap_.find( v );
        if ( it == voxelPathInfoMap_.end() )
        {
            assert( false );
            break;
        }
        auto & vi = it->second;
        if ( vi.isStart() )
            break;
        res.push_back( vi.prevVoxel );
        v = vi.prevVoxel;
    }
    return res;
}

void VoxelsPathsBuilder::addNextStep_( const VoxelPathInfo & c )
{
    auto & vi = voxelPathInfoMap_[c.currVoxel];
    if ( vi.metric > c.metric )
    {
        vi = c;
        nextSteps_.push( c );
    }
}

void VoxelsPathsBuilder::addNeigboursSteps_( float orgMetric, size_t back )
{
    auto coord = getCoord( dimXY_, vdbVolume_.dims.x, back );
    std::vector<size_t> candidates;
    candidates.reserve( 6 );
    if ( coord.x() > 0 )
        candidates.push_back( back - 1 );
    if ( coord.x() < vdbVolume_.dims.x - 1 )
        candidates.push_back( back + 1 );
    if ( coord.y() > 0 )
        candidates.push_back( back - vdbVolume_.dims.x );
    if ( coord.y() < vdbVolume_.dims.y - 1 )
        candidates.push_back( back + vdbVolume_.dims.x );
    if ( coord.z() > 0 )
        candidates.push_back( back - dimXY_ );
    if ( coord.z() < vdbVolume_.dims.z - 1 )
        candidates.push_back( back + dimXY_ );

    for ( const auto& c : candidates )
    {
        VoxelPathInfo cInfo;
        cInfo.currVoxel = c;
        cInfo.prevVoxel = back;
        cInfo.metric = orgMetric + metric_( back, c );
        addNextStep_( cInfo );
    }
}

VoxelsMetric voxelsExponentMetric( const VdbVolume& voxels, const VoxelMetricParameters& parameters, float modifier )
{
    const auto& accessor = voxels.data->getConstAccessor();
    const auto& dims = voxels.dims;
    const int dimsX = dims.x;
    const size_t dimsXY = dimsX * dims.y;
    const auto quaterParams = setupQuaterParams( dimsXY, dimsX, parameters.start, parameters.stop );
    const auto maxDistSq = getDistSq( dimsXY, dimsX, parameters.start, parameters.stop )* parameters.maxDistRatio* parameters.maxDistRatio;
    return [dimsXY, dimsX, accessor, modifier, parameters, maxDistSq, quaterParams]( size_t first, size_t second )
    {
        if ( parameters.plane != None )
        {
            auto nextCoord = getCoord( dimsXY, dimsX, second );
            if ( nextCoord[parameters.plane] != quaterParams.start[parameters.plane] )
                return FLT_MAX;
        }
        if ( !isInQuater( dimsXY, dimsX, quaterParams, second, parameters.quatersMask ) )
            return FLT_MAX;
        if ( ( getDistSq( dimsXY, dimsX, parameters.start, second ) + getDistSq( dimsXY, dimsX, second, parameters.stop ) ) > maxDistSq )
            return FLT_MAX;
        return std::exp( modifier*( accessor.getValue( getCoord( dimsXY, dimsX, first ) ) + accessor.getValue( getCoord( dimsXY, dimsX, second ) ) ) );
    };
}

VoxelsMetric voxelsSumDiffsMetric( const VdbVolume& voxels, const VoxelMetricParameters& parameters )
{
    const auto& accessor = voxels.data->getConstAccessor();
    const auto& dims = voxels.dims;
    const int dimsX = dims.x;
    const size_t dimsXY = dimsX * dims.y;
    const auto quaterParams = setupQuaterParams( dimsXY, dimsX, parameters.start, parameters.stop );
    auto valstart = accessor.getValue( quaterParams.start );
    auto valstop = accessor.getValue( quaterParams.stop );
    const auto maxDistSq = getDistSq( dimsXY, dimsX, parameters.start, parameters.stop )* parameters.maxDistRatio* parameters.maxDistRatio;
    return [dimsXY, dimsX, accessor, valstart, valstop, parameters, maxDistSq, quaterParams]( size_t first, size_t second )
    {
        if ( parameters.plane != None )
        {
            auto nextCoord = getCoord( dimsXY, dimsX, second );
            if ( nextCoord[parameters.plane] != quaterParams.start[parameters.plane] )
                return FLT_MAX;
        }
        if ( !isInQuater( dimsXY, dimsX, quaterParams, second, parameters.quatersMask ) )
            return FLT_MAX;
        if ( ( getDistSq( dimsXY, dimsX, parameters.start, second ) + getDistSq( dimsXY, dimsX, second, parameters.stop ) ) > maxDistSq )
            return FLT_MAX;
        auto val1 = accessor.getValue( getCoord( dimsXY, dimsX, first ) );
        auto val2 = accessor.getValue( getCoord( dimsXY, dimsX, second ) );
        return std::abs( valstart - val1 ) + std::abs( valstop - val1 ) + std::abs( valstart - val2 ) + std::abs( valstop - val2 );
    };
}

std::vector<size_t> buildSmallestMetricPath( const VdbVolume& voxels,  const VoxelsMetric& metric, size_t start, size_t finish )
{
    MR_TIMER;
    VoxelsPathsBuilder b( voxels, metric );
    b.addPathStart( finish, 0 );
    for ( ;;)
    {
        auto back = b.growOneVoxel();
        if ( back == InvalidVoxel )
        {
            // unable to find the path
            return {};
        }
        if ( back == start )
            break;
    }
    return b.getPathBack( start );
}

}
#endif
