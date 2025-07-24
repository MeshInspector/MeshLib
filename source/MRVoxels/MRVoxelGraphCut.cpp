#include "MRVoxelGraphCut.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRTimer.h"
#include "MRVoxelsVolume.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRBox.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <parallel_hashmap/phmap.h>
#include <array>
#include <cfloat>

namespace MR
{

namespace
{

struct VoxelOutEdgeCapacity
{
    float forOutEdge[OutEdgeCount] = { 0, 0, 0, 0, 0, 0 };
};

enum class Side : signed char
{
    Unknown = -1,
    Source = 0,
    Sink
};

inline Side opposite( Side s )
{
    const static std::array<Side, 3> map{ Side::Unknown, Side::Sink, Side::Source };
    return map[ (int)s + 1 ];
}

class VoxelData
{
public:
    // to which side the voxel pertains if any side
    Side side() const { return Side( ( data_ & 3 ) - 1 ); }
    void setSide( Side s ) { data_ = ( data_ & ~3 ) | ( (int)s + 1 ); }

    // out edge pointing to the parent, invalid edge for root faces
    OutEdge parent() const { return OutEdge( ( ( data_ & 28 ) >> 2 ) - 1 ); }
    void setParent( OutEdge e ) { data_ = ( data_ & ~28 ) | ( ( (int)e + 1 ) << 2 ); }

private:
    unsigned char data_ = 0;
};

static_assert( sizeof( VoxelData ) == 1 );

class SeqVoxelTag;
/// sequential id of a voxel in the region of interest (which is typically much smaller then the whole volume)
using SeqVoxelId = Id<SeqVoxelTag>;
using SeqVoxelBitSet = TaggedBitSet<SeqVoxelTag>;

class SpanVoxelTag;
/// id of a voxel inside the span
using SpanVoxelId = Id<SpanVoxelTag>;
using SpanVoxelBitSet = TaggedBitSet<SpanVoxelTag>;

struct SeqVoxelSpan
{
    SeqVoxelId begin, end;
    size_t size() const { return end - begin; }
    SpanVoxelId toSpanId( SeqVoxelId s ) const { return SpanVoxelId( (int)s - (int)begin ); }
    SeqVoxelId  toSeqId( SpanVoxelId p ) const { return SeqVoxelId(  (int)p + (int)begin ); }
    friend bool operator ==( const SeqVoxelSpan &, const SeqVoxelSpan & ) = default;
};

using Neighbors = std::array<SeqVoxelId, OutEdgeCount>;

constexpr int power = 6;
constexpr int numSubtasks = 1 << power;
static_assert( numSubtasks == 64 );

struct ComputedFlow
{
    double outSource = 0; // total flow exiting all sources
    double inSink = 0;    // total flow entering all sinks
};

// cache frequently accessed information about the path toward tree root in a vertex
struct CachePath
{
    SeqVoxelId parent;
    // valid capacities of the edges in the current forest, forward and backward capacities of same edges in capacity_ are outdated
    float capacityToParent = -1;
};
static_assert( sizeof( CachePath ) == 8 );

class VoxelGraphCut : public VolumeIndexer
{
public:
    struct Statistics;
    struct Subtask;
    struct Context;

    using VolumeIndexer::VolumeIndexer;
    /// resizes all data members and fills mappings from original voxel ids to region ids and backward
    void resize( const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds );
    /// returns the span containing all voxels
    SeqVoxelSpan getFullSpan() const { return { SeqVoxelId{ 0 }, seq2voxel_.endId() }; }
    /// returns optimal subdivision of all region voxels on subtasks
    const std::vector<Subtask> & getSubtasks() const { return subtasks_; }
    /// sets edge capacities among all voxel
    void setupCapacities( const SimpleVolume & densityVolume, float k, const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds );
    /// fills neighbor-related data structures
    void setupNeighbors();
    /// removes all references from span-voxels to out-of-span voxels
    void cutOutOfSpanNeiNeighbors( Context & context );
    /// refills neighbor data previously erased by cutOutOfSpanNeiNeighbors
    void restoreCutNeighbor( const Context & context );
    /// constructs forest of paths reaching all voxels in the span
    void buildForest( Context & context, bool initial );
    /// performs min-cut segmentation in given span
    Expected<void> segment( Context & context );
    /// obtain result of segmentation
    VoxelBitSet getResult( const VoxelBitSet & sourceSeeds ) const;
    /// visits all sources/sinks to find the amount of flow
    ComputedFlow computeFlow() const;

private:
    ParallelHashMap<VoxelId, SeqVoxelId> toSeqId_;
    Vector<VoxelId, SeqVoxelId> seq2voxel_;
    std::vector<Subtask> subtasks_;

    Vector<Neighbors, SeqVoxelId> neighbors_;

    Vector<VoxelOutEdgeCapacity, SeqVoxelId> capacity_;
    Vector<VoxelData, SeqVoxelId> voxelData_;
    Vector<CachePath, SeqVoxelId> cachePath_;
    SeqVoxelBitSet sourceSeeds_, sinkSeeds_;
    //std::ofstream f_{R"(D:\logs\voxelgc.txt)"};

    /// allocates all supplementary vectors
    void allocate_( size_t numVoxels );
    /// creates mapping: SeqVoxelId -> VoxelId
    void fillSeq2voxel_( const VoxelBitSet & region );
    /// creates the opposite mapping: VoxelId -> SeqVoxelId
    void fillToSeqId_();
    /// fills neighbors for given voxel
    void setupNeighbors_( SeqVoxelId s );
    // returns ids of all 6 neighbor voxels (or invalid ids if some of them are missing)
    const Neighbors & getNeighbors_( SeqVoxelId s ) const { return neighbors_[s]; }
    // return edge capacity:
    //   from v to vnei for Source side and
    //   from vnei to v for Sink side
    float edgeCapacity_( Side side, SeqVoxelId s, OutEdge vOutEdge, SeqVoxelId seiv ) const;
    // convert given voxel in orphan, writing back cached capacity to/from parent into capacity_ vector
    void addOrphan_( std::vector<SeqVoxelId> & orphans, Side side, SeqVoxelId s, OutEdge edgeToParent, SeqVoxelId sParent, float capacityToParent );
    // process neighborhood of given active voxel
    void processActive_( Context & context, SeqVoxelId s );
    // given a voxel from Unknown side, gives it the best parent; returns false if no parent was found
    bool grow_( SeqVoxelId s );
    // augment the path joined at neighbor voxels vSource and vSink
    void augment_( Context & context, SeqVoxelId sSource, OutEdge vSourceOutEdge, SeqVoxelId sSink );
    // adopt orphans_
    void adopt_( Context & context );
    // tests whether grand is a grandparent of child
    bool isGrandparent_( SeqVoxelId s, SeqVoxelId sGrand ) const;
    // checks that there is not saturated path from f to a root
    bool checkNotSaturatedPath_( SeqVoxelId s, Side side ) const;
    // visits all voxels in given span to find active voxels, where augmentation is necessary;
    // also measures and logs the properties of the current cut
    void findActiveVoxels_( Context & context );
    // creates subtasks including all region voxels
    void makeSubtasks_();
    // given voxels, rearrange them and create subtasks
    void makeSubtasks_( const SeqVoxelSpan & span, Subtask * st, size_t stSize );
    // partition given blocks on two parts, returns the split index in between
    SeqVoxelId partitionVoxels_( const SeqVoxelSpan & span );
    // partition given voxels on two parts using given axis, returns the split index in between
    SeqVoxelId partitionVoxelsByAxis_( const SeqVoxelSpan & span, int axis );
    // computes common bounding box of given voxels
    Box3i computeBbox_( const SeqVoxelSpan & span ) const;
};

struct VoxelGraphCut::Statistics
{
    size_t growths = 0;
    size_t augmentations = 0;
    size_t adoptions = 0;
    size_t grand_calls = 0;
    size_t grand_false = 0;
    double totalFlow = 0;

    // prints this statistics in spdlog
    void log( std::string_view prefix = {} ) const;
    Statistics & operator += ( const Statistics & r );
};

struct VoxelGraphCut::Context
{
    const SeqVoxelSpan span;
    Statistics stat;
    SpanVoxelBitSet cutNeis;
    SpanVoxelBitSet active;
    SpanVoxelBitSet unknown, tmp;
    std::vector<SeqVoxelId> orphans;
    ProgressCallback cb;
};

struct alignas(64) VoxelGraphCut::Subtask
{
    SeqVoxelSpan span;
    VoxelGraphCut::Statistics stat;
};

void VoxelGraphCut::Statistics::log( std::string_view prefix ) const
{
    spdlog::info( "VoxelGraphCut{}: {} augmentations, {} growths, {} adoptions, {} grand_calls, {} grand_false; total flow = {} ",
        prefix, augmentations, growths, adoptions, grand_calls, grand_false, totalFlow );
}

VoxelGraphCut::Statistics & VoxelGraphCut::Statistics::operator += ( const Statistics & r )
{
    growths += r.growths;
    augmentations += r.augmentations;
    adoptions += r.adoptions;
    grand_calls += r.grand_calls;
    grand_false += r.grand_false;
    totalFlow += r.totalFlow;
    return * this;
}

void VoxelGraphCut::allocate_( size_t numVoxels )
{
    MR_TIMER;
    seq2voxel_.reserve( numVoxels );
    toSeqId_.reserve( numVoxels );
    neighbors_.resize( numVoxels );
    voxelData_.resize( numVoxels );
    cachePath_.resize( numVoxels );
    sourceSeeds_.resize( numVoxels );
    sinkSeeds_.resize( numVoxels );
    capacity_.resize( numVoxels );
}

void VoxelGraphCut::fillSeq2voxel_( const VoxelBitSet & region )
{
    MR_TIMER;
    for ( auto vid : region )
        seq2voxel_.push_back( vid );
}

void VoxelGraphCut::fillToSeqId_()
{
    MR_TIMER;
    const auto subcnt = toSeqId_.subcnt();
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subcnt ), [&]( const tbb::blocked_range<size_t> & range )
    {
        assert( range.begin() + 1 == range.end() );
        for ( size_t myPartId = range.begin(); myPartId < range.end(); ++myPartId )
        {
            for ( SeqVoxelId s(0); s < seq2voxel_.size(); ++s )
            {
                auto v = seq2voxel_[s];
                auto hashval = toSeqId_.hash( v );
                auto idx = toSeqId_.subidx( hashval );
                if ( idx != myPartId )
                    continue;
                toSeqId_[v] = s;
            }
        }
    } );
}

void VoxelGraphCut::resize( const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds )
{
    MR_TIMER;

    VoxelBitSet region( size_, true );
    region -= sourceSeeds;
    region -= sinkSeeds;
    const auto cnt0 = region.count();
    expandVoxelsMask( region, *this );
    const auto cnt = region.count();
    const auto srcCnt = sourceSeeds.count();
    const auto snkCnt = sinkSeeds.count();
    spdlog::info( "VoxelGraphCut: {} ({:.3}%) source voxels, {} ({:.3}%) sink voxels, {} ({:.3}%) voxels to classify, {} ({:.3}%) including extra layer",
        srcCnt, float( 100 * srcCnt ) / size_,
        snkCnt, float( 100 * snkCnt ) / size_,
        cnt0, float( 100 * cnt0 ) / size_,
        cnt, float( 100 * cnt ) / size_ );

    allocate_( cnt );
    fillSeq2voxel_( region );
    subtasks_.resize( numSubtasks );
    makeSubtasks_();
    fillToSeqId_();

    assert( size_ == sourceSeeds.size() );
    assert( size_ == sinkSeeds.size() );
    assert( sourceSeeds.any() );
    assert( sinkSeeds.any() );
    assert( !sourceSeeds.intersects( sinkSeeds ) );
    BitSetParallelForAll( sourceSeeds_, [&]( SeqVoxelId s )
    {
        auto v = seq2voxel_[s];
        if ( sourceSeeds.test( v ) )
        {
            voxelData_[s].setSide( Side::Source );
            sourceSeeds_.set( s );
        }
    } );
    BitSetParallelForAll( sinkSeeds_, [&]( SeqVoxelId s )
    {
        auto v = seq2voxel_[s];
        if ( sinkSeeds.test( v ) )
        {
            voxelData_[s].setSide( Side::Sink );
            sinkSeeds_.set( s );
        }
    } );
}

void VoxelGraphCut::setupNeighbors_( SeqVoxelId s )
{
    const auto v = seq2voxel_[s];
    const auto pos = toPos( v );
    const auto bdPos = isBdVoxel( pos );
    for ( int i = 0; i < OutEdgeCount; ++i )
    {
        const auto e = OutEdge( i );
        auto neiv = getNeighbor( v, pos, bdPos, e );
        if ( !neiv )
            continue;
        auto it = toSeqId_.find( neiv );
        if ( it == toSeqId_.end() )
            continue;
        neighbors_[s][i] = it->second;
    }
}

void VoxelGraphCut::setupNeighbors()
{
    MR_TIMER;
    BitSetParallelForAll( sourceSeeds_, [&]( SeqVoxelId s )
    {
        setupNeighbors_( s );
    } );
}

void VoxelGraphCut::restoreCutNeighbor( const Context & context )
{
    MR_TIMER;
    BitSetParallelFor( context.cutNeis, [&]( SpanVoxelId p )
    {
        setupNeighbors_( context.span.toSeqId( p ) );
    } );
}

void VoxelGraphCut::cutOutOfSpanNeiNeighbors( Context & context )
{
    MR_TIMER;
    context.cutNeis.clear();
    context.cutNeis.resize( context.span.size(), false );
    BitSetParallelForAll( context.cutNeis, [&]( SpanVoxelId p )
    {
        const auto s = context.span.toSeqId( p );
        auto & ns = neighbors_[s];
        auto outOfSpan = [&]( SeqVoxelId neis )
        {
            return neis && ( neis < context.span.begin || neis >= context.span.end );
        };
        bool cut = false;
        for ( int i = 0; i < OutEdgeCount; ++i )
        {
            if ( !outOfSpan( ns[i] ) )
                continue;
            cut = true;
            ns[i] = {};
        }
        if ( cut )
            context.cutNeis.set( p );
    } );
}

void VoxelGraphCut::makeSubtasks_()
{
    MR_TIMER;
    makeSubtasks_( getFullSpan(), subtasks_.data(), subtasks_.size() );
}

void VoxelGraphCut::makeSubtasks_( const SeqVoxelSpan & span, Subtask * st, size_t stSize )
{
    assert( stSize >= 1 );
    if ( stSize == 1 )
    {
        std::sort( seq2voxel_.data() + span.begin, seq2voxel_.data() + span.end );
        st[0].span = span;
        return;
    }

    // split subtree between two threads
    const auto spanMid = partitionVoxels_( span );
    const auto stMid = stSize / 2;
    tbb::task_group group;
    group.run( [&] () { makeSubtasks_( { spanMid, span.end }, st + stMid, stSize - stMid ); } );
    makeSubtasks_( { span.begin, spanMid }, st, stMid );
    group.wait();
}

[[maybe_unused]] ComputedFlow VoxelGraphCut::computeFlow() const
{
    MR_TIMER;
    return tbb::parallel_reduce( tbb::blocked_range( SeqVoxelId( 0 ), SeqVoxelId( seq2voxel_.size() ) ), ComputedFlow{},
    [&] ( const tbb::blocked_range<SeqVoxelId> range, ComputedFlow localAcc )
    {
        for ( SeqVoxelId s = range.begin(); s < range.end(); ++s )
        {
            if ( sourceSeeds_.test( s ) )
            {
                const auto & ns = getNeighbors_( s );
                for ( int i = 0; i < OutEdgeCount; ++i )
                    if ( auto neis = ns[i] )
                    {
                        localAcc.outSource += capacity_[neis].forOutEdge[(int)opposite( OutEdge( i ) )]; // exiting flow is equal to entering capacity
                        if ( cachePath_[neis].parent == s )
                            localAcc.outSource += capacity_[s].forOutEdge[i] - cachePath_[neis].capacityToParent;
                    }
            }
            else if ( sinkSeeds_.test( s ) )
            {
                const auto & ns = getNeighbors_( s );
                for ( int i = 0; i < OutEdgeCount; ++i )
                    if ( auto neis = ns[i] )
                    {
                        localAcc.inSink += capacity_[s].forOutEdge[i]; // entering flow is equal to exiting capacity
                        if ( cachePath_[neis].parent == s )
                            localAcc.inSink += capacity_[neis].forOutEdge[(int)opposite( OutEdge( i ) )] - cachePath_[neis].capacityToParent;
                    }
            }
        }
        return localAcc;
    },
    [&] ( ComputedFlow a, const ComputedFlow& b )
    {
        a.outSource += b.outSource;
        a.inSink += b.inSink;
        return a;
    } );
}

Box3i VoxelGraphCut::computeBbox_( const SeqVoxelSpan & span ) const
{
    return tbb::parallel_reduce( tbb::blocked_range( span.begin, span.end ), Box3i{},
    [&] ( const tbb::blocked_range<SeqVoxelId> range, Box3i localAcc )
    {
        for ( SeqVoxelId i = range.begin(); i < range.end(); ++i )
            localAcc.include( toPos( seq2voxel_[i] ) );
        return localAcc;
    },
    [&] ( Box3i a, const Box3i& b )
    {
        a.include( b );
        return a;
    } );
}

SeqVoxelId VoxelGraphCut::partitionVoxels_( const SeqVoxelSpan & span )
{
    const auto box = computeBbox_( span );
    auto boxDiag = box.max - box.min;
    const int splitAxis = int( std::max_element( begin( boxDiag ), end( boxDiag ) ) - begin( boxDiag ) );
    return partitionVoxelsByAxis_( span, splitAxis );
}

SeqVoxelId VoxelGraphCut::partitionVoxelsByAxis_( const SeqVoxelSpan & span, int axis )
{
    auto mid = span.begin + ( span.end - span.begin ) / 2;
    mid = SeqVoxelId( mid / BitSet::bits_per_block * BitSet::bits_per_block );
    assert( mid >= span.begin && mid < span.end );

    switch ( axis )
    {
    case 0: // X
        std::nth_element( seq2voxel_.data() + span.begin, seq2voxel_.data() + mid, seq2voxel_.data() + span.end,
            [&]( VoxelId a, VoxelId b ) { return ( a % dims_.x ) < ( b % dims_.x ); } );
        break;
    case 1: // Y
        std::nth_element( seq2voxel_.data() + span.begin, seq2voxel_.data() + mid, seq2voxel_.data() + span.end,
            [&]( VoxelId a, VoxelId b ) { return ( a % sizeXY_ ) < ( b % sizeXY_ ); } );
        break;
    case 2: // Z
        std::nth_element( seq2voxel_.data() + span.begin, seq2voxel_.data() + mid, seq2voxel_.data() + span.end,
            [&]( VoxelId a, VoxelId b ) { return a < b; } );
        break;
    default:
        assert( false );
    }
    return mid;
}

void VoxelGraphCut::setupCapacities( const SimpleVolume & densityVolume, float k, const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds )
{
    MR_TIMER;

    // prevent infinite capacities
    constexpr float maxCapacity = FLT_MAX / 10;
    const float maxDelta = log( maxCapacity ) / std::abs( k );
    //spdlog::info( "maxDelta={}", maxDelta );

    auto capacity = [=]( float densityFrom, float densityTo )
    {
        const auto delta = densityTo - densityFrom;
        if ( ( k > 0 && delta > maxDelta ) || ( k < 0 && delta < -maxDelta ) )
            return maxCapacity;
        return std::exp( k * delta );
    };

    tbb::parallel_for( tbb::blocked_range<SeqVoxelId>( SeqVoxelId( 0 ), SeqVoxelId( seq2voxel_.size() ) ), [&] ( const tbb::blocked_range<SeqVoxelId>& range )
    {
        for ( SeqVoxelId sid = range.begin(); sid != range.end(); ++sid )
        {
            if ( sinkSeeds_.test( sid ) )
                continue; // no exiting edges from sinks
            auto & cap = capacity_[sid];
            const auto vid = seq2voxel_[sid];
            const auto density = densityVolume.data[vid];
            const auto pos = toPos( vid );
            const auto bdPos = isBdVoxel( pos );
            const bool vIsSource = sourceSeeds_.test( sid );
            for ( int i = 0; i < OutEdgeCount; ++i )
            {
                const auto e = OutEdge( i );
                auto neiv = getNeighbor( vid, pos, bdPos, e );
                if ( !neiv )
                    continue;
                if ( sourceSeeds.test( neiv ) )
                    continue; // no entering edges to sources
                if ( vIsSource && sinkSeeds.test( neiv ) )
                    continue; // no direct source-sink edges
                cap.forOutEdge[i] = capacity( density, densityVolume.data[neiv] );
            }
        }
    } );
}

void VoxelGraphCut::buildForest( Context & context, bool initial )
{
    MR_TIMER;

    context.unknown.resize( context.span.size(), true );
    assert( ( context.span.begin % BitSet::bits_per_block ) == 0 );
    static_cast<BitSet&>( context.unknown ).subtract( sourceSeeds_, -(int)context.span.begin / BitSet::bits_per_block );
    static_cast<BitSet&>( context.unknown ).subtract( sinkSeeds_, -(int)context.span.begin / BitSet::bits_per_block );

    if ( !initial )
    {
        BitSetParallelFor( context.unknown, [&]( SpanVoxelId p )
        {
            const auto s = context.span.toSeqId( p );
            if ( voxelData_[s].side() != Side::Unknown )
                context.unknown.reset( p );
        } );
    }

    context.tmp = context.unknown;
    [[maybe_unused]] int layers = 0;
    for ( ;; ++layers )
    {
        //spdlog::info( "VoxelGraphCut layer #{}: {} voxels to fill", layers, cnt );
        std::atomic<bool> changed{false};
        BitSetParallelFor( context.unknown, [&]( SpanVoxelId p )
        {
            const auto s = context.span.toSeqId( p );
            auto & vd = voxelData_[s];
            assert( vd.side() == Side::Unknown );
            Side bestSide = Side::Unknown;
            OutEdge bestParentEdge = OutEdge::Invalid;
            SeqVoxelId bestParent;
            float bestCapacity = 0;
            const auto & ns = getNeighbors_( s );
            for ( int i = 0; i < OutEdgeCount; ++i )
            {
                auto neis = ns[i];
                if ( !neis )
                    continue;
                if ( context.unknown.test( context.span.toSpanId( neis ) ) )
                    continue;
                const auto neiSide = voxelData_[neis].side();
                assert ( neiSide != Side::Unknown );
                const auto e = OutEdge( i );
                auto capacity = edgeCapacity_( neiSide, neis, opposite( e ), s );
                assert( capacity >= 0 );
                if ( capacity > 0 && ( bestCapacity == 0 || capacity < bestCapacity ) )
                {
                    bestCapacity = capacity;
                    bestParentEdge = e;
                    bestParent = neis;
                    bestSide = neiSide;
                }
            }
            if ( bestParent )
            {
                vd.setSide( bestSide );
                vd.setParent( bestParentEdge );
                cachePath_[s] = CachePath{ bestParent, bestCapacity };
                context.tmp.reset( p );
                bool expected = false;
                changed.compare_exchange_strong( expected, true );
            }
        } );
        if ( !changed )
            break;
        context.unknown = context.tmp;
    }
    //spdlog::info( "VoxelGraphCut: {} initial layers", layers );
}

Expected<void> VoxelGraphCut::segment( Context & context )
{
    MR_TIMER;

    findActiveVoxels_( context );

    float progress = 1.0f / 16;
    if ( !reportProgress( context.cb, progress ) )
        return unexpectedOperationCanceled();
    const float targetProgress = 1.0f;

    for ( size_t i = 0;; ++i )
    {
        constexpr size_t STEP = 1024ull * 1024;
        if ( context.cb && ( i % STEP == 0 ) )
        {
            progress += ( targetProgress - progress ) * 0.5f;
            if ( !context.cb( progress ) )
                return unexpectedOperationCanceled();
        }
        bool anyActive = false;
        for ( auto p : context.active )
        {
            context.active.reset( p );
            processActive_( context, context.span.toSeqId( p ) );
            anyActive = true;
        }
        if ( !anyActive )
            break;
    }
    if ( context.span == getFullSpan() ) // just for final logging
        findActiveVoxels_( context );

    return {};
}

VoxelBitSet VoxelGraphCut::getResult( const VoxelBitSet & sourceSeeds ) const
{
    MR_TIMER;
    VoxelBitSet res = sourceSeeds;
    for ( SeqVoxelId s( 0 ); s < seq2voxel_.size(); ++s )
        if ( voxelData_[s].side() == Side::Source )
            res.set( seq2voxel_[s] );
    const auto resCnt = res.count();
    spdlog::info( "VoxelGraphCut result: {} ({:.3}%) source voxels, {} ({:.3}%) sink voxels",
        resCnt, float( 100 * resCnt ) /size_,
        size_ - resCnt, float( 100 * ( size_ - resCnt ) ) /size_ );
    return res;
}

void VoxelGraphCut::findActiveVoxels_( Context & context )
{
    MR_TIMER;

    assert( ( context.span.begin % BitSet::bits_per_block ) == 0 );
    context.active.resize( context.span.size() );

    BitSetParallelForAll( IdRange<SeqVoxelId>{ context.span.begin, context.span.end }, [&]( SeqVoxelId s )
    {
        const auto side = voxelData_[s].side();
        if ( side == Side::Unknown )
            return;
        const auto & ns = getNeighbors_( s );
        for ( int i = 0; i < OutEdgeCount; ++i )
        {
            auto neis = ns[i];
            if ( !neis )
                continue;
            const auto neiSide = voxelData_[neis].side();
            if ( side == neiSide || ( side == Side::Sink && neiSide == Side::Source ) )
                continue;
            const auto e = OutEdge( i );
            const float edgeCapacity = edgeCapacity_( side, s, e, neis );
            assert( edgeCapacity >= 0 );
            if ( edgeCapacity > 0 )
            {
                context.active.set( context.span.toSpanId( s ) );
                break;
            }
        }
    } );

    if ( context.span == getFullSpan() )
    {
        size_t numVoxels[3] = {};
        for ( SeqVoxelId s( 0 ); s < seq2voxel_.size(); ++s )
        {
            const auto side = voxelData_[s].side();
            ++numVoxels[(int)side+1];
        }

        size_t totalCutEdges = 0;
        size_t unsaturatedCutEdges = 0;
        double cutCapacity = 0;
        for ( auto p : context.active )
        {
            const auto s = context.span.toSeqId( p );
            const auto side = voxelData_[s].side();
            assert( side != Side::Unknown );
            const auto & ns = getNeighbors_( s );
            for ( int i = 0; i < OutEdgeCount; ++i )
            {
                auto neis = ns[i];
                if ( !neis )
                    continue;
                const auto neiSide = voxelData_[neis].side();
                if ( side == neiSide || ( side == Side::Sink && neiSide == Side::Source ) )
                    continue;
                ++totalCutEdges;
                const auto e = OutEdge( i );
                const float edgeCapacity = edgeCapacity_( side, s, e, neis );
                assert( edgeCapacity >= 0 );
                if ( edgeCapacity > 0 )
                {
                    ++unsaturatedCutEdges;
                    cutCapacity += edgeCapacity;
                }
            }
        }
        spdlog::info( "VoxelGraphCut region: {} unknown voxels, {} source voxels, {} sink voxels, {} active voxels", numVoxels[0], numVoxels[1], numVoxels[2], context.active.count() );
        spdlog::info( "VoxelGraphCut cut: {} total cut edges, {} unsaturated cut edges, cut capacity = {}", totalCutEdges, unsaturatedCutEdges, cutCapacity );
    }
}

inline float VoxelGraphCut::edgeCapacity_( Side side, SeqVoxelId s, OutEdge vOutEdge, SeqVoxelId neis ) const
{
    assert( s && neis );
    assert( getNeighbor( seq2voxel_[s], vOutEdge ) == seq2voxel_[neis] );
    assert( side != Side::Unknown );
    if ( side == Side::Source )
        return capacity_[s].forOutEdge[(int)vOutEdge];
    else
        return capacity_[neis].forOutEdge[(int)opposite( vOutEdge )];
}

void VoxelGraphCut::addOrphan_( std::vector<SeqVoxelId> & orphans, Side side, SeqVoxelId s, OutEdge edgeToParent, SeqVoxelId sParent, float capacityToParent )
{
    assert( s && sParent );
    assert( cachePath_[s].parent == sParent );
    assert( voxelData_[s].side() == side );
    assert( voxelData_[s].parent() == edgeToParent );
    assert( cachePath_[s].capacityToParent == capacityToParent );
    assert( capacityToParent >= 0 );
    if ( side == Side::Source )
    {
        auto & orgCapacity = capacity_[ sParent ].forOutEdge[ (int)opposite( edgeToParent ) ];
        assert( orgCapacity >= capacityToParent );
        capacity_[ s ].forOutEdge[ (int) edgeToParent ] += ( orgCapacity - capacityToParent );
        orgCapacity = capacityToParent;
    }
    else
    {
        assert( side == Side::Sink );
        auto & orgCapacity = capacity_[ s ].forOutEdge[ (int) edgeToParent ];
        assert( orgCapacity >= capacityToParent );
        capacity_[ sParent ].forOutEdge[ (int)opposite( edgeToParent ) ] += ( orgCapacity - capacityToParent );
        orgCapacity = capacityToParent;
    }
#ifndef NDEBUG
    cachePath_[s].capacityToParent = -1;
#endif
    voxelData_[s].setParent( OutEdge::Invalid );
    cachePath_[s].parent = {};
    orphans.push_back( s );
}

void VoxelGraphCut::processActive_( Context & context, SeqVoxelId s )
{
    const auto & vd = voxelData_[s];
    const auto side = vd.side();
    if ( vd.side() == Side::Unknown )
        return; // voxel has changed the side since the moment it was put in the queue

    auto edgeToParent = vd.parent();
    const auto & ns = getNeighbors_( s );
    for ( int i = 0; i < OutEdgeCount; ++i )
    {
        const auto e = OutEdge( i );
        if ( e == edgeToParent )
            continue;
        auto neis = ns[i];
        if ( !neis )
            continue;
        auto & neid = voxelData_[neis];
        if ( neid.side() == opposite( side ) )
        {
            if ( side == Side::Source )
                augment_( context, s, e, neis );
            else
                augment_( context, neis, opposite( e ), s );
            if ( vd.side() != side )
                return; // voxel has changed the side during augmentation
            continue;
        }
        if ( neid.side() == side )
            continue;
        if ( grow_( neis ) )
        {
            context.active.set( context.span.toSpanId( neis ) );
            ++context.stat.growths;
        }
    }
}

bool VoxelGraphCut::grow_( SeqVoxelId s )
{
    assert( s );
    auto & vd = voxelData_[s];
    assert( vd.side() == Side::Unknown );
    assert( vd.parent() == OutEdge::Invalid );
    assert( cachePath_[s].capacityToParent < 0 );
    Side bestSide = Side::Unknown;
    OutEdge bestParentEdge = OutEdge::Invalid;
    SeqVoxelId bestParent;
    float bestCapacity = 0;
    const auto & ns = getNeighbors_( s );
    for ( int i = 0; i < OutEdgeCount; ++i )
    {
        auto neis = ns[i];
        if ( !neis )
            continue;
        const auto neiSide = voxelData_[neis].side();
        if ( neiSide == Side::Unknown )
            continue;
        const auto e = OutEdge( i );
        auto capacity = edgeCapacity_( neiSide, neis, opposite( e ), s );
        assert( capacity >= 0 );
        if ( capacity > 0 && ( bestCapacity == 0 || capacity < bestCapacity ) )
        {
            bestCapacity = capacity;
            bestParentEdge = e;
            bestParent = neis;
            bestSide = neiSide;
        }
    }
    if ( bestParent )
    {
        vd.setSide( bestSide );
        vd.setParent( bestParentEdge );
        cachePath_[s] = CachePath{ bestParent, bestCapacity };
        assert( checkNotSaturatedPath_( s, bestSide ) );
        return true;
    }
    return false;
}

void VoxelGraphCut::augment_( Context & context, SeqVoxelId sSource, OutEdge vSourceOutEdge, SeqVoxelId sSink )
{
    assert( sSource && sSink );
    assert( getNeighbor( seq2voxel_[sSource], vSourceOutEdge ) == seq2voxel_[sSink] );
    auto & srcD = voxelData_[sSource];
    auto & snkD = voxelData_[sSink];

    // zero means "need update"
    float sourceCapacity = 0;
    float sinkCapacity = 0;
    for ( ;; )
    {
        assert( srcD.side() == Side::Source );
        assert( snkD.side() == Side::Sink );
        assert( checkNotSaturatedPath_( sSource, Side::Source ) );
        assert( checkNotSaturatedPath_( sSink, Side::Sink ) );

        const auto joinCapacity = capacity_[ sSource ].forOutEdge[ (int)vSourceOutEdge ];
        assert( joinCapacity >= 0 );
        if ( joinCapacity == 0 )
            break;
        ++context.stat.augmentations;

        if ( sourceCapacity <= 0 )
        {
            sourceCapacity = FLT_MAX;
            for ( auto s = sSource;; )
            {
                assert( voxelData_[s].side() == Side::Source );
                auto sParent = cachePath_[s].parent;
                if ( !sParent )
                    break;
                sourceCapacity = std::min( sourceCapacity, cachePath_[s].capacityToParent );
                assert( sourceCapacity > 0 );
                s = sParent;
            }
        }
        if ( sinkCapacity <= 0 )
        {
            sinkCapacity = FLT_MAX;
            for ( auto s = sSink;; )
            {
                assert( voxelData_[s].side() == Side::Sink );
                auto sParent = cachePath_[s].parent;
                if ( !sParent )
                    break;
                sinkCapacity = std::min( sinkCapacity, cachePath_[s].capacityToParent );
                assert( sinkCapacity > 0 );
                s = sParent;
            }
        }

        auto minResidualCapacity = std::min( { sinkCapacity, sourceCapacity, joinCapacity } );
        assert( minResidualCapacity >= 0 );
        if ( minResidualCapacity == 0 )
            break;

        assert( minResidualCapacity > 0 );
        capacity_[ sSource ].forOutEdge[ (int)vSourceOutEdge ] -= minResidualCapacity;
        capacity_[ sSink ].forOutEdge[ (int)opposite( vSourceOutEdge ) ] += minResidualCapacity;
        sourceCapacity -= minResidualCapacity;
        sinkCapacity -= minResidualCapacity;
        context.stat.totalFlow += minResidualCapacity;
        //f_ << totalFlow_ << '\t' << minResidualCapacity << '\n';

        assert( context.orphans.empty() );
        for ( auto s = sSource;; )
        {
            assert( voxelData_[s].side() == Side::Source );
            auto sParent = cachePath_[s].parent;
            if ( !sParent )
                break;
            if ( ( cachePath_[s].capacityToParent -= minResidualCapacity ) == 0 )
                addOrphan_( context.orphans, Side::Source, s, voxelData_[s].parent(), sParent, 0 );
            s = sParent;
        }

        for ( auto s = sSink;; )
        {
            assert( voxelData_[s].side() == Side::Sink );
            auto sParent = cachePath_[s].parent;
            if ( !sParent )
                break;
            if ( ( cachePath_[s].capacityToParent -= minResidualCapacity ) == 0 )
                addOrphan_( context.orphans, Side::Sink, s, voxelData_[s].parent(), sParent, 0 );
            s = sParent;
        }
        adopt_( context );

        if ( srcD.side() != Side::Source || snkD.side() != Side::Sink )
            break;
    }
}

#if __GNUC__ >= 12 //https://gcc.gnu.org/bugzilla/show_bug.cgi?id=104165
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

void VoxelGraphCut::adopt_( Context & context )
{
    while ( !context.orphans.empty() )
    {
        const auto s = context.orphans.back();
        auto & vd = voxelData_[s];
        context.orphans.pop_back();
        const auto side = vd.side();
        assert( side != Side::Unknown );
        assert( vd.parent() == OutEdge::Invalid );
        assert( cachePath_[s].capacityToParent < 0 );
        OutEdge bestParentEdge = OutEdge::Invalid, bestOtherSideParentEdge = OutEdge::Invalid;
        SeqVoxelId bestParent, bestOtherSideParent;
        float bestCapacity = 0, bestOtherSideCapacity = 0;
        bool directChild[OutEdgeCount] = {};
        int directChildren = 0;
        struct OutCapacity
        {
            float c = 0;
            OutEdge e = OutEdge::Invalid;
            auto operator <=>( const OutCapacity & ) const = default;
        };
        std::array<OutCapacity, OutEdgeCount> outCapacity;
        int numOut = 0;
        bool grandChild[OutEdgeCount] = {};
        const auto & ns = getNeighbors_( s );
        for ( int i = 0; i < OutEdgeCount; ++i )
        {
            const auto neis = ns[i];
            if ( !neis )
                continue;
            const auto & neid = voxelData_[neis];
            if ( neid.side() == Side::Unknown )
                continue;
            const auto e = OutEdge( i );
            if ( opposite( e ) == neid.parent() )
            {
                assert( neid.side() == side );
                directChild[i] = true;
                ++directChildren;
                continue;
            }
            if ( neid.side() == side )
            {
                float capacity = edgeCapacity_( side, neis, opposite( e ), s );
                assert( capacity >= 0 );
                if ( capacity > 0 )
                    outCapacity[numOut++] = OutCapacity{ capacity, e };
            }
            else
            {
                float capacity = edgeCapacity_( opposite( side ), neis, opposite( e ), s );
                assert( capacity >= 0 );
                if ( capacity > 0 && ( bestOtherSideCapacity == 0 || capacity < bestOtherSideCapacity ) )
                {
                    bestOtherSideParentEdge = e;
                    bestOtherSideParent = neis;
                    bestOtherSideCapacity = capacity;
                }
            }
        }
        #if __GNUC__ == 14
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wstringop-overflow"
        #endif
        std::sort( outCapacity.begin(), outCapacity.begin() + numOut );
        #if __GNUC__ == 14
        #pragma GCC diagnostic pop
        #endif
        for ( int j = 0; j < numOut; ++j )
        {
            const auto & o = outCapacity[j];
            const auto neis = ns[(int)o.e];
            if ( directChildren > 0 )
            {
                ++context.stat.grand_calls;
                if ( isGrandparent_( neis, s ) )
                {
                    grandChild[(int)o.e] = true;
                    continue;
                }
                ++context.stat.grand_false;
            }
            bestParentEdge = o.e;
            bestParent = neis;
            bestCapacity = o.c;
            break;
        }
        if ( bestParent )
        {
            ++context.stat.adoptions;
            vd.setParent( bestParentEdge );
            cachePath_[s] = CachePath{ bestParent, bestCapacity };
            assert( checkNotSaturatedPath_( s, side ) );
            context.active.set( context.span.toSpanId( s ) );
        }
        else
        {
            for ( int i = 0; i < OutEdgeCount; ++i )
            {
                if ( !directChild[i] )
                    continue;
                const auto neis = ns[i];
                addOrphan_( context.orphans, side, neis, opposite( OutEdge( i ) ), s, cachePath_[neis].capacityToParent );
            }
            if ( bestOtherSideParent )
            {
                ++context.stat.growths;
                vd.setSide( opposite( side ) );
                vd.setParent( bestOtherSideParentEdge );
                cachePath_[s] = CachePath{ bestOtherSideParent, bestOtherSideCapacity };
                assert( checkNotSaturatedPath_( s, opposite( side ) ) );
                context.active.set( context.span.toSpanId( s ) );
            }
            else
            {
                vd.setSide( Side::Unknown );
                vd.setParent( OutEdge::Invalid );
                cachePath_[s].parent = {};
                for ( int i = 0; i < OutEdgeCount; ++i )
                {
                    // after finishing of adoption, grandChild can become a new parent for this voxel
                    if ( grandChild[i] )
                        context.active.set( context.span.toSpanId( ns[i] ) );
                }
            }
        }
    }
}

inline bool VoxelGraphCut::isGrandparent_( SeqVoxelId s, SeqVoxelId sGrand ) const
{
    assert( s && sGrand );
    while ( s != sGrand )
    {
        s = cachePath_[s].parent;
        if ( !s )
            return false;
    }
    return true;
}

[[maybe_unused]] bool VoxelGraphCut::checkNotSaturatedPath_( SeqVoxelId s, [[maybe_unused]] Side side ) const
{
    assert( side != Side::Unknown );
    for ( ;; )
    {
        assert( voxelData_[s].side() == side );
        auto sParent = cachePath_[s].parent;
        if ( !sParent )
            return true;
        assert( cachePath_[s].capacityToParent > 0 );
        s = sParent;
    }
}

} // anonymous namespace

Expected<VoxelBitSet> segmentVolumeByGraphCut( const SimpleVolume & densityVolume, float k, const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds, ProgressCallback cb )
{
    MR_TIMER;

    if ( !reportProgress( cb, 0.0f ) )
        return unexpectedOperationCanceled();

    VoxelGraphCut vgc( densityVolume.dims );
    vgc.resize( sourceSeeds, sinkSeeds );
    if ( !reportProgress( cb, 2.0f / 16 ) )
        return unexpectedOperationCanceled();

    vgc.setupCapacities( densityVolume, k, sourceSeeds, sinkSeeds );
    if ( !reportProgress( cb, 3.0f / 16 ) )
        return unexpectedOperationCanceled();

    vgc.setupNeighbors();
    if ( !reportProgress( cb, 4.0f / 16 ) )
        return unexpectedOperationCanceled();

    auto parts = vgc.getSubtasks();
    // parallel threads shall be able to safely modify elements in bit-sets
    const auto voxelsPerPart = ( int( vgc.getFullSpan().end ) / ( numSubtasks * BitSet::bits_per_block ) ) * BitSet::bits_per_block;

    auto sp = subprogress( cb, 4.0f / 16, 1.0f );
    for ( int p = 0; p <= power; ++p )
    {
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, parts.size() ), [&]( const tbb::blocked_range<size_t>& range )
        {
            for ( size_t i = range.begin(); i < range.end(); ++i )
            {
                auto & part = parts[i];
                if ( parts.size() == numSubtasks )
                {
                    part.span.begin = SeqVoxelId( i * voxelsPerPart );
                    part.span.end = ( i + 1 ) < numSubtasks ? SeqVoxelId( ( i + 1 ) * voxelsPerPart ) : vgc.getFullSpan().end;
                }

                VoxelGraphCut::Context context
                {
                    .span = part.span,
                    .stat = part.stat
                };
                if ( parts.size() > 1 )
                    vgc.cutOutOfSpanNeiNeighbors( context );
                vgc.buildForest( context, parts.size() == numSubtasks );
                (void)vgc.segment( context ); // cannot fail without callback
                if ( parts.size() > 1 )
                    vgc.restoreCutNeighbor( context );
                part.stat = context.stat;
                //part.stat.log( fmt::format( " after [{}, {})", part.span.begin, part.span.end ) );
            }
        } );
        if ( parts.size() <= 1 )
        {
            parts[0].stat.log( " final" );
            break;
        }
        VoxelGraphCut::Statistics total;
        for ( size_t i = 0; 2 * i + 1 < parts.size(); ++i )
        {
            auto p0 = parts[ 2 * i ];
            auto p1 = parts[ 2 * i + 1 ];
            assert( p0.span.end == p1.span.begin );
            p0.span.end = p1.span.end;
            p0.stat += p1.stat;
            //p0.stat.log( fmt::format( " before [{}, {})", p0.span.begin, p0.span.end ) );
            parts[i] = p0;
            total += p0.stat;
        }
        total.log( fmt::format( " after {} parts", parts.size() ) );
        parts.resize( parts.size() / 2 );
        if ( !reportProgress( cb, 1.0f / ( power + 1 ) ) )
            return unexpectedOperationCanceled();
    }
    //auto cflow = vgc.computeFlow();
    //spdlog::info( "VoxelGraphCut: flow-exiting-sources={}, flow-entering-sinks={}", cflow.outSource, cflow.inSink );
    return vgc.getResult( sourceSeeds );
}

} // namespace MR
