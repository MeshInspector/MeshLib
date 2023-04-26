#include "MRVoxelGraphCut.h"
#include "MRVector.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRSimpleVolume.h"
#include "MRVolumeIndexer.h"
#include "MRBitSetParallelFor.h"
#include "MRHash.h"
#include "MRExpected.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <parallel_hashmap/phmap.h>
#include <array>

namespace MR
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

struct SeqVoxelSpan
{
    SeqVoxelId begin, end;
    friend bool operator ==( const SeqVoxelSpan &, const SeqVoxelSpan & ) = default;
};

struct YZNeigbours
{
    SeqVoxelId minusZ, plusZ;
    std::uint16_t minusY, plusY;
};
static_assert( sizeof( YZNeigbours ) == 12 );

using Neighbours = std::array<SeqVoxelId, OutEdgeCount>;

class VoxelGraphCut : public VolumeIndexer
{
public:
    struct Statistics;

    using VolumeIndexer::VolumeIndexer;
    /// resizes all data members and fills mappings from original voxel ids to region ids and backward
    void resize( const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds );
    /// returns the span containing all voxels
    SeqVoxelSpan getFullSpan() const { return { SeqVoxelId{ 0 }, seq2voxel_.endId() }; }
    /// sets edge capacities among all voxel
    void setupCapacities( const SimpleVolume & densityVolume, float k );
    /// fills neighbor-related data structures
    void setupNeighbors();
    /// removes all references from span-voxels to out-of-span voxels
    void cutOutOfSpanNeiNeighbors( const SeqVoxelSpan & span );
    /// refills neighbor data previously erased by cutOutOfSpanNeiNeighbors
    void restoreCutNeighbor();
    /// constructs initial forest of paths reaching all voxels in the span
    void buildInitialForest( const SeqVoxelSpan & span );
    /// performs min-cut segmentation in given span
    VoidOrErrStr segment( const SeqVoxelSpan & span, Statistics & stat, ProgressCallback cb = {} );
    /// obtain result of segmentation
    VoxelBitSet getResult( const VoxelBitSet & sourceSeeds ) const;

private:
    phmap::flat_hash_map<VoxelId, SeqVoxelId> toSeqId_;
    Vector<VoxelId, SeqVoxelId> seq2voxel_;

    // neighbors:
    BitSet xNeighbors_;
    Vector<YZNeigbours, SeqVoxelId> yzNeighbors_;

    Vector<VoxelOutEdgeCapacity, SeqVoxelId> capacity_;
    Vector<VoxelData, SeqVoxelId> voxelData_;
    Vector<SeqVoxelId, SeqVoxelId> parent_;
    // valid capacities of the edges in the current forest, forward and backward capacities of same edges in capacity_ are outdated
    Vector<float, SeqVoxelId> capacityToParent_;
    SeqVoxelBitSet sourceSeeds_, sinkSeeds_;
    SeqVoxelBitSet active_, cutNeis_;
    //std::ofstream f_{R"(D:\logs\voxelgc.txt)"};

    // allocates all supplementary vectors
    void allocate_( size_t numVoxels );
    /// fills neighbors for given voxel
    void setupNeighbors_( SeqVoxelId s );
    // returns ids of all 6 neighbor voxels (or invalid ids if some of them are missing)
    Neighbours getNeighbors_( SeqVoxelId s ) const;
    // return edge capacity:
    //   from v to vnei for Source side and
    //   from vnei to v for Sink side
    float edgeCapacity_( Side side, SeqVoxelId s, OutEdge vOutEdge, SeqVoxelId seiv ) const;
    // convert given voxel in orphan, writing back cached capacity to/from parent into capacity_ vector
    void addOrphan_( std::vector<SeqVoxelId> & orphans, Side side, SeqVoxelId s, OutEdge edgeToParent, SeqVoxelId sParent, float capacityToParent );
    // process neighborhood of given active voxel
    void processActive_( SeqVoxelId s, std::vector<SeqVoxelId> & orphans, Statistics & stat );
    // given a voxel from Unknown side, gives it the best parent; returns false if no parent was found
    bool grow_( SeqVoxelId s );
    // augment the path joined at neighbor voxels vSource and vSink
    void augment_( SeqVoxelId sSource, OutEdge vSourceOutEdge, SeqVoxelId sSink, std::vector<SeqVoxelId> & orphans, Statistics & stat );
    // adopt orphans_
    void adopt_( std::vector<SeqVoxelId> & orphans, Statistics & stat );
    // tests whether grand is a grandparent of child
    bool isGrandparent_( SeqVoxelId s, SeqVoxelId sGrand ) const;
    // checks that there is not saturated path from f to a root
    bool checkNotSaturatedPath_( SeqVoxelId s, Side side ) const;
    // visits all voxels in given span to find active voxels, where augmentation is necessary;
    // also measures and logs the properties of the current cut
    void findActiveVoxels_( const SeqVoxelSpan & span );
};

struct VoxelGraphCut::Statistics
{
    size_t growths = 0;
    size_t augmentations = 0;
    size_t adoptions = 0;
    double totalFlow = 0;

    // prints this statistics in spdlog
    void log( std::string_view prefix = {} ) const;
    Statistics & operator += ( const Statistics & r );
};

void VoxelGraphCut::Statistics::log( std::string_view prefix ) const
{
    spdlog::info( "VoxelGraphCut{}: {} augmentations, {} growths, {} adoptions; total flow = {} ", 
        prefix, augmentations, growths, adoptions, totalFlow );
}

VoxelGraphCut::Statistics & VoxelGraphCut::Statistics::operator += ( const Statistics & r )
{
    growths += r.growths;
    augmentations += r.augmentations;
    adoptions += r.adoptions;
    totalFlow += r.totalFlow;
    return * this;
}

void VoxelGraphCut::allocate_( size_t numVoxels )
{
    MR_TIMER
    seq2voxel_.reserve( numVoxels );
    toSeqId_.reserve( numVoxels );
    xNeighbors_.resize( 2 * numVoxels );
    yzNeighbors_.resize( numVoxels );
    voxelData_.resize( numVoxels );
    parent_.resize( numVoxels );
    capacityToParent_.resize( numVoxels );
    sourceSeeds_.resize( numVoxels );
    sinkSeeds_.resize( numVoxels );
    active_.resize( numVoxels, false );
    cutNeis_.resize( numVoxels, false );
    capacity_.resize( numVoxels );
}

void VoxelGraphCut::resize( const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds )
{
    MR_TIMER

    VoxelBitSet region;
    region.resize( size_, true );
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
    for ( auto vid : region )
    {
        toSeqId_[ vid ] = SeqVoxelId( seq2voxel_.size() );
        seq2voxel_.push_back( vid );
    }

    assert( size_ == sourceSeeds.size() );
    assert( size_ == sinkSeeds.size() );
    assert( sourceSeeds.any() );
    assert( sinkSeeds.any() );
    assert( ( sourceSeeds & sinkSeeds ).count() == 0 );
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
        const auto neis = it->second;
        switch ( e )
        {
        case MR::OutEdge::PlusZ:
            yzNeighbors_[s].plusZ = neis;
            break;
        case MR::OutEdge::MinusZ:
            yzNeighbors_[s].minusZ = neis;
            break;
        case MR::OutEdge::PlusY:
        {
            auto delta = neis - s;
            assert( delta > 0 && delta < 65536 );
            yzNeighbors_[s].plusY = std::uint16_t( delta );
            break;
        }
        case MR::OutEdge::MinusY:
        {
            auto delta = s - neis;
            assert( delta > 0 && delta < 65536 );
            yzNeighbors_[s].minusY = std::uint16_t( delta );
            break;
        }
        case MR::OutEdge::PlusX:
            assert( neis == s + 1 );
            xNeighbors_.set( 2 * s + 1 );
            break;
        case MR::OutEdge::MinusX:
            assert( neis == s - 1 );
            xNeighbors_.set( 2 * s );
            break;
        default:
            assert( false );
        }
    }
}

void VoxelGraphCut::setupNeighbors()
{
    MR_TIMER
    BitSetParallelForAll( cutNeis_, [&]( SeqVoxelId s )
    {
        setupNeighbors_( s );
    } );
}

void VoxelGraphCut::restoreCutNeighbor()
{
    MR_TIMER
    BitSetParallelFor( cutNeis_, [&]( SeqVoxelId s )
    {
        setupNeighbors_( s );
        cutNeis_.reset( s );
    } );
}

void VoxelGraphCut::cutOutOfSpanNeiNeighbors( const SeqVoxelSpan & span )
{
    MR_TIMER
    BitSetParallelForAll( span.begin, span.end, [&]( SeqVoxelId s )
    {
        const auto ns = getNeighbors_( s );
        auto outOfSpan = [&]( SeqVoxelId neis )
        {
            return neis && ( neis < span.begin || neis >= span.end );
        };
        bool cut = false;
        if ( outOfSpan( ns[(int)MR::OutEdge::PlusZ] ) )
            cut = true, yzNeighbors_[s].plusZ = {};
        if ( outOfSpan( ns[(int)MR::OutEdge::MinusZ] ) )
            cut = true, yzNeighbors_[s].minusZ = {};
        if ( outOfSpan( ns[(int)MR::OutEdge::PlusY] ) )
            cut = true, yzNeighbors_[s].plusY = 0;
        if ( outOfSpan( ns[(int)MR::OutEdge::MinusY] ) )
            cut = true, yzNeighbors_[s].minusY = 0;
        if ( outOfSpan( ns[(int)MR::OutEdge::PlusX] ) )
            cut = true, xNeighbors_.reset( 2 * s + 1 );
        if ( outOfSpan( ns[(int)MR::OutEdge::MinusX] ) )
            cut = true, xNeighbors_.reset( 2 * s );
        if ( cut )
            cutNeis_.set( s );
    } );
}

inline Neighbours VoxelGraphCut::getNeighbors_( SeqVoxelId s ) const
{
    auto dMinusY = yzNeighbors_[s].minusY;
    auto dPlusY = yzNeighbors_[s].plusY;
    return Neighbours {
        yzNeighbors_[s].plusZ,
        yzNeighbors_[s].minusZ,
        dPlusY ? SeqVoxelId( s + dPlusY ) : SeqVoxelId{},
        dMinusY ? SeqVoxelId( s - dMinusY ) : SeqVoxelId{},
        xNeighbors_.test( 2 * s + 1 ) ? SeqVoxelId( s + 1 ) : SeqVoxelId{},
        xNeighbors_.test( 2 * s ) ? SeqVoxelId( s - 1 ) : SeqVoxelId{}
    };
}

void VoxelGraphCut::setupCapacities( const SimpleVolume & densityVolume, float k )
{
    MR_TIMER

    // prevent infinite capacities
    constexpr float maxCapacity = FLT_MAX / 10;
    const float maxDelta = log( maxCapacity ) / std::abs( k );
    //spdlog::info( "maxDelta={}", maxDelta );

    auto capacity = [=]( float densityFrom, float densityTo )
    {
        const auto delta = densityTo - densityFrom;
        if ( delta > maxDelta )
            return maxCapacity;
        return std::exp( k * delta );
    };

    tbb::parallel_for( tbb::blocked_range<SeqVoxelId>( SeqVoxelId( 0 ), SeqVoxelId( seq2voxel_.size() ) ), [&] ( const tbb::blocked_range<SeqVoxelId>& range )
    {
        for ( SeqVoxelId sid = range.begin(); sid != range.end(); ++sid )
        {
            auto & cap = capacity_[sid];
            const auto vid = seq2voxel_[sid];
            auto density = densityVolume.data[vid];
            auto pos = toPos( vid );
            if ( pos.x > 0 )
                cap.forOutEdge[( int )OutEdge::MinusX] = capacity( density, densityVolume.data[vid - size_t( 1 )] );
            if ( pos.x + 1 < dims_.x )
                cap.forOutEdge[( int )OutEdge::PlusX] = capacity( density, densityVolume.data[vid + size_t( 1 )] );
            if ( pos.y > 0 )
                cap.forOutEdge[( int )OutEdge::MinusY] = capacity( density, densityVolume.data[vid - size_t( dims_.x )] );
            if ( pos.y + 1 < dims_.y )
                cap.forOutEdge[( int )OutEdge::PlusY] = capacity( density, densityVolume.data[vid + size_t( dims_.x )] );
            if ( pos.z > 0 )
                cap.forOutEdge[ (int)OutEdge::MinusZ ] = capacity( density, densityVolume.data[ vid - sizeXY_ ] );
            if ( pos.z + 1 < dims_.z )
                cap.forOutEdge[ (int)OutEdge::PlusZ ] = capacity( density, densityVolume.data[ vid + sizeXY_ ] );
        }
    } );
}

void VoxelGraphCut::buildInitialForest( const SeqVoxelSpan & span )
{
    MR_TIMER

    SeqVoxelBitSet toFill;
    toFill.resize( span.end, false );
    toFill.set( span.begin, span.end - span.begin, true );
    toFill -= sourceSeeds_;
    toFill -= sinkSeeds_;
    SeqVoxelBitSet nextToFill = toFill;
    [[maybe_unused]] int layers = 0;
    auto cnt = toFill.count();
    for ( ;; ++layers )
    {
        //spdlog::info( "VoxelGraphCut layer #{}: {} voxels to fill", layers, cnt );
        BitSetParallelFor( toFill, [&]( SeqVoxelId s )
        {
            auto & vd = voxelData_[s];
            assert( vd.side() == Side::Unknown );
            Side bestSide = Side::Unknown;
            OutEdge bestParentEdge = OutEdge::Invalid;
            SeqVoxelId bestParent;
            float bestCapacity = 0;
            const auto ns = getNeighbors_( s );
            for ( int i = 0; i < OutEdgeCount; ++i )
            {
                auto neis = ns[i];
                if ( !neis )
                    continue;
                if ( toFill.test( neis ) )
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
                parent_[s] = bestParent;
                capacityToParent_[s] = bestCapacity;
                nextToFill.reset( s );
            }
        } );
        auto cnt1 = nextToFill.count();
        if ( cnt == cnt1 )
            break;
        toFill = nextToFill;
        cnt = cnt1;
    }
    //spdlog::info( "VoxelGraphCut: {} initial layers", layers );
}

VoidOrErrStr VoxelGraphCut::segment( const SeqVoxelSpan & span, Statistics & stat, ProgressCallback cb )
{
    MR_TIMER
    
    findActiveVoxels_( span );
    
    float progress = 1.0f / 16;
    if ( !reportProgress( cb, progress ) )
        return tl::make_unexpected( "Operation was canceled" );
    const float targetProgress = 1.0f;
    
    std::vector<SeqVoxelId> orphans;
    for ( size_t i = 0;; ++i )
    {
        constexpr size_t STEP = 1024ull * 1024;
        if ( cb && ( i % STEP == 0 ) )
        {
            progress += ( targetProgress - progress ) * 0.5f;
            if ( !cb( progress ) )
                return tl::make_unexpected( "Operation was canceled" );
        }
        bool anyActive = false;
        for ( auto s = ( span.begin == 0 ) ? active_.find_first() : active_.find_next( span.begin );
              s && s < span.end;
              s = active_.find_next( s ) )
        {
            active_.reset( s );
            processActive_( s, orphans, stat );
            anyActive = true;
        }
        if ( !anyActive )
            break;
    }
    findActiveVoxels_( span );

    return {};
}

VoxelBitSet VoxelGraphCut::getResult( const VoxelBitSet & sourceSeeds ) const
{
    MR_TIMER
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

void VoxelGraphCut::findActiveVoxels_( const SeqVoxelSpan & span )
{
    MR_TIMER

    BitSetParallelForAll( span.begin, span.end, [&]( SeqVoxelId s )
    {
        const auto side = voxelData_[s].side();
        if ( side == Side::Unknown )
            return;
        const bool vIsSeed = sourceSeeds_.test( s ) || sinkSeeds_.test( s );
        const auto ns = getNeighbors_( s );
        for ( int i = 0; i < OutEdgeCount; ++i )
        {
            auto neis = ns[i];
            if ( !neis )
                continue;
            const auto neiSide = voxelData_[neis].side();
            if ( side == neiSide || ( side == Side::Sink && neiSide == Side::Source ) )
                continue;
            if ( vIsSeed && ( sourceSeeds_.test( neis ) || sinkSeeds_.test( neis ) ) )
                continue;
            const auto e = OutEdge( i );
            const float edgeCapacity = edgeCapacity_( side, s, e, neis );
            assert( edgeCapacity >= 0 );
            if ( edgeCapacity > 0 )
            {
                active_.set( s );
                break;
            }
        }
    } );

    if ( span == getFullSpan() )
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
        for ( auto s : active_ )
        {
            const auto side = voxelData_[s].side();
            assert( side != Side::Unknown );
            const bool vIsSeed = sourceSeeds_.test( s ) || sinkSeeds_.test( s );
            const auto ns = getNeighbors_( s );
            for ( int i = 0; i < OutEdgeCount; ++i )
            {
                auto neis = ns[i];
                if ( !neis )
                    continue;
                const auto neiSide = voxelData_[neis].side();
                if ( side == neiSide || ( side == Side::Sink && neiSide == Side::Source ) )
                    continue;
                if ( vIsSeed && ( sourceSeeds_.test( neis ) || sinkSeeds_.test( neis ) ) )
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
        spdlog::info( "VoxelGraphCut region: {} unknown voxels, {} source voxels, {} sink voxels, {} active voxels", numVoxels[0], numVoxels[1], numVoxels[2], active_.count() );
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
    assert( parent_[s] == sParent );
    assert( voxelData_[s].side() == side );
    assert( voxelData_[s].parent() == edgeToParent );
    assert( capacityToParent_[s] == capacityToParent );
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
    capacityToParent_[s] = -100;
#endif
    voxelData_[s].setParent( OutEdge::Invalid );
    parent_[s] = {};
    orphans.push_back( s );
}

void VoxelGraphCut::processActive_( SeqVoxelId s, std::vector<SeqVoxelId> & orphans, Statistics & stat )
{
    const auto & vd = voxelData_[s];
    const auto side = vd.side();
    if ( vd.side() == Side::Unknown )
        return; // voxel has changed the side since the moment it was put in the queue

    auto edgeToParent = vd.parent();
    const auto ns = getNeighbors_( s );
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
                augment_( s, e, neis, orphans, stat );
            else
                augment_( neis, opposite( e ), s, orphans, stat );
            if ( vd.side() != side )
                return; // voxel has changed the side during augmentation
            continue;
        }
        if ( neid.side() == side )
            continue;
        if ( grow_( neis ) )
           ++stat.growths;
    }
}

bool VoxelGraphCut::grow_( SeqVoxelId s )
{
    assert( s );
    auto & vd = voxelData_[s];
    assert( vd.side() == Side::Unknown );
    assert( vd.parent() == OutEdge::Invalid );
    assert( capacityToParent_[s] < 0 );
    Side bestSide = Side::Unknown;
    OutEdge bestParentEdge = OutEdge::Invalid;
    SeqVoxelId bestParent;
    float bestCapacity = 0;
    const auto ns = getNeighbors_( s );
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
        parent_[s] = bestParent;
        capacityToParent_[s] = bestCapacity;
        assert( checkNotSaturatedPath_( s, bestSide ) );
        active_.set( s );
        return true;
    }
    return false;
}

void VoxelGraphCut::augment_( SeqVoxelId sSource, OutEdge vSourceOutEdge, SeqVoxelId sSink, std::vector<SeqVoxelId> & orphans, Statistics & stat )
{
    assert( sSource && sSink );
    assert( getNeighbor( seq2voxel_[sSource], vSourceOutEdge ) == seq2voxel_[sSink] );
    if ( sourceSeeds_.test( sSource ) && sinkSeeds_.test( sSink ) )
        return;
    auto & srcD = voxelData_[sSource];
    auto & snkD = voxelData_[sSink];

    for ( ;; )
    {
        assert( srcD.side() == Side::Source );
        assert( snkD.side() == Side::Sink );
        assert( checkNotSaturatedPath_( sSource, Side::Source ) );
        assert( checkNotSaturatedPath_( sSink, Side::Sink ) );

        auto minResidualCapacity = capacity_[ sSource ].forOutEdge[ (int)vSourceOutEdge ];
        assert( minResidualCapacity >= 0 );
        if ( minResidualCapacity == 0 )
            break;
        ++stat.augmentations;

        for ( auto s = sSource;; )
        {
            assert( voxelData_[s].side() == Side::Source );
            auto sParent = parent_[s];
            if ( !sParent )
                break;
            minResidualCapacity = std::min( minResidualCapacity, capacityToParent_[s] );
            s = sParent;
        }
        for ( auto s = sSink;; )
        {
            assert( voxelData_[s].side() == Side::Sink );
            auto sParent = parent_[s];
            if ( !sParent )
                break;
            minResidualCapacity = std::min( minResidualCapacity, capacityToParent_[s] );
            s = sParent;
        }

        assert( minResidualCapacity > 0 );
        capacity_[ sSource ].forOutEdge[ (int)vSourceOutEdge ] -= minResidualCapacity;
        capacity_[ sSink ].forOutEdge[ (int)opposite( vSourceOutEdge ) ] += minResidualCapacity;
        stat.totalFlow += minResidualCapacity;
        //f_ << totalFlow_ << '\t' << minResidualCapacity << '\n';

        assert( orphans.empty() );
        for ( auto s = sSource;; )
        {
            assert( voxelData_[s].side() == Side::Source );
            auto sParent = parent_[s];
            if ( !sParent )
                break;
            if ( ( capacityToParent_[s] -= minResidualCapacity ) == 0 )
                addOrphan_( orphans, Side::Source, s, voxelData_[s].parent(), sParent, 0 );
            s = sParent;
        }

        for ( auto s = sSink;; )
        {
            assert( voxelData_[s].side() == Side::Sink );
            auto sParent = parent_[s];
            if ( !sParent )
                break;
            if ( ( capacityToParent_[s] -= minResidualCapacity ) == 0 )
                addOrphan_( orphans, Side::Sink, s, voxelData_[s].parent(), sParent, 0 );
            s = sParent;
        }
        adopt_( orphans, stat );

        if ( srcD.side() != Side::Source || snkD.side() != Side::Sink )
            break;
    }
}

void VoxelGraphCut::adopt_( std::vector<SeqVoxelId> & orphans, Statistics & stat )
{
    while ( !orphans.empty() )
    {
        const auto s = orphans.back();
        auto & vd = voxelData_[s];
        orphans.pop_back();
        const auto side = vd.side();
        assert( side != Side::Unknown );
        assert( vd.parent() == OutEdge::Invalid );
        assert( capacityToParent_[s] < 0 );
        OutEdge bestParentEdge = OutEdge::Invalid, bestOtherSideParentEdge = OutEdge::Invalid;
        SeqVoxelId bestParent, bestOtherSideParent;
        float bestCapacity = 0, bestOtherSideCapacity = 0;
        bool directChild[OutEdgeCount] = {};
        bool grandChild[OutEdgeCount] = {};
        const auto ns = getNeighbors_( s );
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
                continue;
            }
            if ( neid.side() == side )
            {
                float capacity = edgeCapacity_( side, neis, opposite( e ), s );
                assert( capacity >= 0 );
                if ( capacity > 0 && ( bestCapacity == 0 || capacity < bestCapacity ) )
                {
                    if ( isGrandparent_( neis, s ) )
                        grandChild[i] = true;
                    else
                    {
                        bestParentEdge = e;
                        bestParent = neis;
                        bestCapacity = capacity;
                    }
                }
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
        if ( bestParent )
        {
            ++stat.adoptions;
            vd.setParent( bestParentEdge );
            parent_[s] = bestParent;
            capacityToParent_[s] = bestCapacity;
            assert( checkNotSaturatedPath_( s, side ) );
            active_.set( s );
        }
        else
        {
            for ( int i = 0; i < OutEdgeCount; ++i )
            {
                if ( !directChild[i] )
                    continue;
                const auto neis = ns[i];
                addOrphan_( orphans, side, neis, opposite( OutEdge( i ) ), s, capacityToParent_[neis] );
            }
            if ( bestOtherSideParent )
            {
                ++stat.growths;
                vd.setSide( opposite( side ) );
                vd.setParent( bestOtherSideParentEdge );
                parent_[s] = bestOtherSideParent;
                capacityToParent_[s] = bestOtherSideCapacity;
                assert( checkNotSaturatedPath_( s, opposite( side ) ) );
                active_.set( s );
            }
            else
            {
                vd.setSide( Side::Unknown );
                vd.setParent( OutEdge::Invalid );
                parent_[s] = {};
                for ( int i = 0; i < OutEdgeCount; ++i )
                {
                    // after finishing of adoption, grandChild can become a new parent for this voxel
                    if ( grandChild[i] )
                        active_.set( ns[i] );
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
        s = parent_[s];
        if ( !s )
            return false;
    }
    return true;
}

bool VoxelGraphCut::checkNotSaturatedPath_( SeqVoxelId s, [[maybe_unused]] Side side ) const
{
    assert( side != Side::Unknown );
    for ( ;; )
    {
        assert( voxelData_[s].side() == side );
        auto sParent = parent_[s];
        if ( !sParent )
            return true;
        assert( capacityToParent_[s] > 0 );
        s = sParent;
    }
}

tl::expected<VoxelBitSet, std::string> segmentVolumeByGraphCut( const SimpleVolume & densityVolume, float k, const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds, ProgressCallback cb )
{
    MR_TIMER

    if ( !reportProgress( cb, 0.0f ) )
        return tl::make_unexpected( "Operation was canceled" );

    VoxelGraphCut vgc( densityVolume.dims );
    vgc.resize( sourceSeeds, sinkSeeds );
    if ( !reportProgress( cb, 2.0f / 16 ) )
        return tl::make_unexpected( "Operation was canceled" );

    vgc.setupCapacities( densityVolume, k );
    if ( !reportProgress( cb, 3.0f / 16 ) )
        return tl::make_unexpected( "Operation was canceled" );

    vgc.setupNeighbors();
    if ( !reportProgress( cb, 4.0f / 16 ) )
        return tl::make_unexpected( "Operation was canceled" );

    constexpr int power = 6;
    constexpr int numParts = 1 << power;
    static_assert( numParts == 64 );
    struct alignas(64) Part
    {
        SeqVoxelSpan span;
        VoxelGraphCut::Statistics stat;
    };
    std::vector<Part> parts( numParts );
    // parallel threads shall be able to safely modify elements in bit-sets
    const auto voxelsPerPart = ( int( vgc.getFullSpan().end ) / ( numParts * BitSet::bits_per_block ) ) * BitSet::bits_per_block;

    auto sp = subprogress( cb, 4.0f / 16, 1.0f );
    for ( int p = 0; p <= power; ++p )
    {
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, parts.size() ), [&]( const tbb::blocked_range<size_t>& range )
        {
            for ( size_t i = range.begin(); i < range.end(); ++i )
            {
                auto & part = parts[i];
                if ( parts.size() == numParts )
                {
                    part.span.begin = SeqVoxelId( i * voxelsPerPart );
                    part.span.end = ( i + 1 ) < numParts ? SeqVoxelId( ( i + 1 ) * voxelsPerPart ) : vgc.getFullSpan().end;
                }

                if ( parts.size() > 1 )
                    vgc.cutOutOfSpanNeiNeighbors( part.span );
                if ( parts.size() == numParts )
                    vgc.buildInitialForest( part.span );
                vgc.segment( part.span, part.stat );
                //part.stat.log( fmt::format( " after [{}, {})", part.span.begin, part.span.end ) );
            }
        } );
        if ( parts.size() <= 1 )
        {
            parts[0].stat.log( " final" );
            break;
        }
        vgc.restoreCutNeighbor();
        VoxelGraphCut::Statistics total;
        for ( size_t i = 0; 2 * i + 1 < parts.size(); ++i )
        {
            Part p0 = parts[ 2 * i ];
            Part p1 = parts[ 2 * i + 1 ];
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
            return tl::make_unexpected( "Operation was canceled" );
    }
    return vgc.getResult( sourceSeeds );
}

} // namespace MR
