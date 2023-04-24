#include "MRVoxelGraphCut.h"
#include "MRVector.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRSimpleVolume.h"
#include "MRVolumeIndexer.h"
#include "MRBitSetParallelFor.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"

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

class VoxelGraphCut : public VolumeIndexer
{
public:
    using VolumeIndexer::VolumeIndexer;
    tl::expected<VoxelBitSet, std::string> fill( const SimpleVolume & densityVolume, float k,
        const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds, ProgressCallback cb );

private:
    Vector<SeqVoxelId, VoxelId> voxel2seq_;
    Vector<VoxelId, SeqVoxelId> seq2voxel_;
    Vector<VoxelOutEdgeCapacity, SeqVoxelId> capacity_;
    Vector<VoxelData, SeqVoxelId> voxelData_;
    SeqVoxelBitSet sourceSeeds_, sinkSeeds_;
    SeqVoxelBitSet active_;
    std::vector<SeqVoxelId> orphans_;
    //statistics:
    size_t growths_ = 0;
    size_t augmentations_ = 0;
    size_t adoptions_ = 0;
    double totalFlow_ = 0;
    //std::ofstream f_{R"(D:\logs\voxelgc.txt)"};

    // return edge capacity:
    //   from v to vnei for Source side and
    //   from vnei to v for Sink side
    float edgeCapacity_( Side side, SeqVoxelId s, OutEdge vOutEdge, SeqVoxelId seiv ) const;
    // resizes all data members and fills voxel2seq_, seq2voxel_
    void fillVoxel2seq_( const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds );
    // fill capacity_
    void fillCapacities_( const SimpleVolume & densityVolume, float k );
    // constructs initial forest of paths processing vertices in min-edge-capacity-in-path-till-vertex order
    bool buildInitialForest_();
    // process neighborhood of given active voxel
    void processActive_( SeqVoxelId s );
    // given a voxel from Unknown side, gives it the best parent
    void grow_( SeqVoxelId s );
    // augment the path joined at neighbor voxels vSource and vSink
    void augment_( SeqVoxelId sSource, OutEdge vSourceOutEdge, SeqVoxelId sSink );
    // adopt orphans_
    void adopt_();
    // tests whether grand is a grandparent of child
    bool isGrandparent_( SeqVoxelId s, SeqVoxelId sGrand ) const;
    // checks that there is not saturated path from f to a root
    bool checkNotSaturatedPath_( SeqVoxelId s, Side side ) const;
    // print current statistics in spdlog
    void logStatistics_() const;
    // visits all voxels to find active voxels, where augmentation is necessary;
    // also measures and logs the properties of the current cut
    void findActiveVoxels_();
};

void VoxelGraphCut::fillVoxel2seq_( const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds )
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

    seq2voxel_.reserve( cnt );
    voxel2seq_.resize( size_ );
    for ( auto vid : region )
{
        voxel2seq_[ vid ] = SeqVoxelId( seq2voxel_.size() );
        seq2voxel_.push_back( vid );
    }
    voxelData_.resize( seq2voxel_.size() );
    sourceSeeds_.resize( seq2voxel_.size() );
    sinkSeeds_.resize( seq2voxel_.size() );

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

void VoxelGraphCut::fillCapacities_( const SimpleVolume & densityVolume, float k )
{
    MR_TIMER

    capacity_.resize( seq2voxel_.size() );

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

bool VoxelGraphCut::buildInitialForest_()
{
    MR_TIMER

    SeqVoxelBitSet toFill;
    toFill.resize( seq2voxel_.size(), true );
    toFill -= sourceSeeds_;
    toFill -= sinkSeeds_;
    SeqVoxelBitSet nextToFill = toFill;
    int layers = 0;
    auto cnt = toFill.count();
    for ( ;; ++layers )
    {
        //spdlog::info( "VoxelGraphCut layer #{}: {} voxels to fill", layers, cnt );
        BitSetParallelFor( toFill, [&]( SeqVoxelId s )
    {
            auto & vd = voxelData_[s];
            assert( vd.side() == Side::Unknown );
        const auto v = seq2voxel_[s];
        const auto pos = toPos( v );
        const auto bdPos = isBdVoxel( pos );
            Side bestSide = Side::Unknown;
            OutEdge bestParent = OutEdge::Invalid;
            float bestCapacity = 0;
        for ( auto e : all6Edges )
        {
            auto neiv = getNeighbor( v, pos, bdPos, e );
            if ( !neiv )
                continue;
            auto neis = voxel2seq_[neiv];
            if ( !neis )
                continue;
                if ( toFill.test( neis ) )
                    continue;
                const auto neiSide = voxelData_[neis].side();
                assert ( neiSide != Side::Unknown );
                auto capacity = edgeCapacity_( neiSide, neis, opposite( e ), s );
                if ( capacity > 0 && ( bestCapacity == 0 || capacity < bestCapacity ) )
            {
                    bestCapacity = capacity;
                    bestParent = e;
                    bestSide = neiSide;
                }
            }
            if ( bestParent != OutEdge::Invalid )
            {
                vd.setSide( bestSide );
                vd.setParent( bestParent );
                nextToFill.reset( s );
            }
        } );
        auto cnt1 = nextToFill.count();
        if ( cnt == cnt1 )
            break;
        toFill = nextToFill;
        cnt = cnt1;
        }
    spdlog::info( "VoxelGraphCut: {} initial layers", layers );

    return true;
}

tl::expected<VoxelBitSet, std::string> VoxelGraphCut::fill( const SimpleVolume & densityVolume, float k,
    const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds, ProgressCallback cb )
{
    MR_TIMER
    
    if ( !reportProgress( cb, 0.0f ) )
        return tl::make_unexpected( "Operation was canceled" );

    fillVoxel2seq_( sourceSeeds, sinkSeeds );
    if ( !reportProgress( cb, 1.0f / 6 ) )
        return tl::make_unexpected( "Operation was canceled" );

    fillCapacities_( densityVolume, k );
    if ( !reportProgress( cb, 2.0f / 6 ) )
        return tl::make_unexpected( "Operation was canceled" );
    
    buildInitialForest_();
    if ( !reportProgress( cb, 3.0f / 6 ) )
        return tl::make_unexpected( "Operation was canceled" );
    findActiveVoxels_();
    
    float progress = 0.5f;
    const float targetProgress = 1.0f;
    
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
        for ( auto s : active_ )
        {
            active_.reset( s );
            processActive_( s );
            anyActive = true;
        }
        if ( !anyActive )
            break;
    }
    logStatistics_();
    findActiveVoxels_();

    if ( cb && !cb( targetProgress ) )
        return tl::make_unexpected( "Operation was canceled" );

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

void VoxelGraphCut::logStatistics_() const
{
    spdlog::info( "VoxelGraphCut: {} augmentations, {} growths, {} adoptions; total flow = {} ", augmentations_, growths_, adoptions_, totalFlow_ );
}

void VoxelGraphCut::findActiveVoxels_()
{
    MR_TIMER

    active_.clear();
    active_.resize( seq2voxel_.size(), false );
    BitSetParallelForAll( active_, [&]( SeqVoxelId s )
    {
        const auto side = voxelData_[s].side();
        if ( side != Side::Source )
            return;
        const auto v = seq2voxel_[s];
        const auto pos = toPos( v );
        const auto bdPos = isBdVoxel( pos );
        const bool vIsSourceSeed = sourceSeeds_.test( s );
        for ( auto e : all6Edges )
        {
            auto neiv = getNeighbor( v, pos, bdPos, e );
            if ( !neiv )
                continue;
            auto neis = voxel2seq_[neiv];
            if ( !neis )
                continue;
            if ( voxelData_[neis].side() != Side::Sink )
                continue;
            if ( vIsSourceSeed && sinkSeeds_.test( neis ) )
                continue;
            const float edgeCapacity = edgeCapacity_( Side::Source, s, e, neis );
            if ( edgeCapacity > 0 )
            {
                active_.set( s );
                break;
            }
        }
    } );

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
        assert ( voxelData_[s].side() == Side::Source );
        const auto v = seq2voxel_[s];
        const auto pos = toPos( v );
        const auto bdPos = isBdVoxel( pos );
        const bool vIsSourceSeed = sourceSeeds_.test( s );
        for ( auto e : all6Edges )
        {
            auto neiv = getNeighbor( v, pos, bdPos, e );
            if ( !neiv )
                continue;
            auto neis = voxel2seq_[neiv];
            if ( !neis )
                continue;
            if ( voxelData_[neis].side() != Side::Sink )
                continue;
            if ( vIsSourceSeed && sinkSeeds_.test( neis ) )
                continue;
            ++totalCutEdges;
            const float edgeCapacity = edgeCapacity_( Side::Source, s, e, neis );
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

void VoxelGraphCut::processActive_( SeqVoxelId s )
{
    const auto & vd = voxelData_[s];
    const auto side = vd.side();
    if ( vd.side() == Side::Unknown )
        return; // voxel has changed the side since the moment it was put in the queue

    auto v = seq2voxel_[s];
    const auto pos = toPos( v );
    const auto bdPos = isBdVoxel( pos );
    auto edgeToParent = vd.parent();

    for ( auto e : all6Edges )
    {
        if ( e == edgeToParent )
            continue;
        auto neiv = getNeighbor( v, pos, bdPos, e );
        if ( !neiv )
            continue;
        auto neis = voxel2seq_[neiv];
        if ( !neis )
            continue;
        auto & neid = voxelData_[neis];
        if ( neid.side() == opposite( side ) )
        {
            if ( side == Side::Source )
                augment_( s, e, neis );
            else
                augment_( neis, opposite( e ), s );
            if ( vd.side() != side )
                return; // voxel has changed the side during augmentation
            continue;
        }
        if ( neid.side() == side )
            continue;
        grow_( neis );
    }
}

void VoxelGraphCut::grow_( SeqVoxelId s )
{
    assert( s );
    auto & vd = voxelData_[s];
    assert( vd.side() == Side::Unknown );
    const auto v = seq2voxel_[s];
    const auto pos = toPos( v );
    const auto bdPos = isBdVoxel( pos );
    Side bestSide = Side::Unknown;
    OutEdge bestParent = OutEdge::Invalid;
    float bestCapacity = 0;
    for ( auto e : all6Edges )
    {
        auto neiv = getNeighbor( v, pos, bdPos, e );
        if ( !neiv )
            continue;
        auto neis = voxel2seq_[neiv];
        if ( !neis )
            continue;
        const auto neiSide = voxelData_[neis].side();
        if ( neiSide == Side::Unknown )
            continue;
        auto capacity = edgeCapacity_( neiSide, neis, opposite( e ), s );
        if ( capacity > 0 && ( bestCapacity == 0 || capacity < bestCapacity ) )
        {
            bestCapacity = capacity;
            bestParent = e;
            bestSide = neiSide;
        }
    }
    if ( bestParent != OutEdge::Invalid )
    {
        ++growths_;
        vd.setSide( bestSide );
        vd.setParent( bestParent );
        assert( checkNotSaturatedPath_( s, bestSide ) );
        active_.set( s );
    }
}

void VoxelGraphCut::augment_( SeqVoxelId sSource, OutEdge vSourceOutEdge, SeqVoxelId sSink )
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
        ++augmentations_;
        constexpr size_t STEP = 1024ull * 1024;
        if ( augmentations_ % STEP == 0 )
            logStatistics_();

        for ( auto s = sSource;; )
        {
            assert( voxelData_[s].side() == Side::Source );
            auto edgeToParent = voxelData_[s].parent();
            if ( edgeToParent == OutEdge::Invalid )
                break;
            const auto v = seq2voxel_[s];
            auto vParent = getExistingNeighbor( v, edgeToParent );
            auto sParent = voxel2seq_[vParent];
            assert( sParent );
            minResidualCapacity = std::min( minResidualCapacity, capacity_[ sParent ].forOutEdge[ (int)opposite( edgeToParent ) ] );
            s = sParent;
        }
        for ( auto s = sSink;; )
        {
            assert( voxelData_[s].side() == Side::Sink );
            auto edgeToParent = voxelData_[s].parent();
            if ( edgeToParent == OutEdge::Invalid )
                break;
            const auto v = seq2voxel_[s];
            auto vParent = getExistingNeighbor( v, edgeToParent );
            auto sParent = voxel2seq_[vParent];
            assert( sParent );
            minResidualCapacity = std::min( minResidualCapacity, capacity_[ s ].forOutEdge[ (int) edgeToParent ] );
            s = sParent;
        }

        assert( minResidualCapacity > 0 );
        capacity_[ sSource ].forOutEdge[ (int)vSourceOutEdge ] -= minResidualCapacity;
        capacity_[ sSink ].forOutEdge[ (int)opposite( vSourceOutEdge ) ] += minResidualCapacity;
        totalFlow_ += minResidualCapacity;
        //f_ << totalFlow_ << '\t' << minResidualCapacity << '\n';

        assert( orphans_.empty() );
        for ( auto s = sSource;; )
        {
            assert( voxelData_[s].side() == Side::Source );
            auto edgeToParent = voxelData_[s].parent();
            if ( edgeToParent == OutEdge::Invalid )
                break;
            const auto v = seq2voxel_[s];
            auto vParent = getExistingNeighbor( v, edgeToParent );
            auto sParent = voxel2seq_[vParent];
            assert( sParent );
            capacity_[ s ].forOutEdge[ (int) edgeToParent ] += minResidualCapacity;
            if ( ( capacity_[ sParent ].forOutEdge[ (int)opposite( edgeToParent ) ] -= minResidualCapacity ) == 0 )
            {
                voxelData_[s].setParent( OutEdge::Invalid );
                orphans_.push_back( s );
            }
            s = sParent;
        }

        for ( auto s = sSink;; )
        {
            assert( voxelData_[s].side() == Side::Sink );
            auto edgeToParent = voxelData_[s].parent();
            if ( edgeToParent == OutEdge::Invalid )
                break;
            const auto v = seq2voxel_[s];
            auto vParent = getExistingNeighbor( v, edgeToParent );
            auto sParent = voxel2seq_[vParent];
            assert( sParent );
            capacity_[ sParent ].forOutEdge[ (int)opposite( edgeToParent ) ] += minResidualCapacity;
            if ( ( capacity_[ s ].forOutEdge[ (int) edgeToParent ] -= minResidualCapacity ) == 0 )
            {
                voxelData_[s].setParent( OutEdge::Invalid );
                orphans_.push_back( s );
            }
            s = sParent;
        }
        adopt_();

        if ( srcD.side() != Side::Source || snkD.side() != Side::Sink )
            break;
    }
}

void VoxelGraphCut::adopt_()
{
    while ( !orphans_.empty() )
    {
        const auto s = orphans_.back();
        auto & vd = voxelData_[s];
        orphans_.pop_back();
        const auto side = vd.side();
        assert( side != Side::Unknown );
        assert( vd.parent() == OutEdge::Invalid );
        const auto v = seq2voxel_[s];
        const auto pos = toPos( v );
        const auto bdPos = isBdVoxel( pos );
        OutEdge bestParent = OutEdge::Invalid, bestOtherSideParent = OutEdge::Invalid;
        float bestCapacity = 0, bestOtherSideCapacity = 0;
        bool directChild[OutEdgeCount] = {};
        for ( auto e : all6Edges )
        {
            auto neiv = getNeighbor( v, pos, bdPos, e );
            if ( !neiv )
                continue;
            const auto neis = voxel2seq_[neiv];
            if ( !neis )
                continue;
            const auto & neid = voxelData_[neis];
            if ( neid.side() == Side::Unknown )
                continue;
            if ( opposite( e ) == neid.parent() )
            {
                assert( neid.side() == side );
                directChild[(int)e] = true;
                continue;
            }
            if ( neid.side() == side )
            {
                float capacity = edgeCapacity_( side, neis, opposite( e ), s );
                if ( capacity > 0 && ( bestCapacity == 0 || capacity < bestCapacity ) && !isGrandparent_( neis, s ) )
                {
                    bestParent = e;
                    bestCapacity = capacity;
                }
            }
            else
            {
                float capacity = edgeCapacity_( opposite( side ), neis, opposite( e ), s );
                if ( capacity > 0 && ( bestOtherSideCapacity == 0 || capacity < bestOtherSideCapacity ) )
                {
                    bestOtherSideParent = e;
                    bestOtherSideCapacity = capacity;
                }
            }
        }
        if ( bestParent != OutEdge::Invalid )
        {
            ++adoptions_;
            vd.setParent( bestParent );
            assert( checkNotSaturatedPath_( s, side ) );
            active_.set( s );
        }
        else
        {
            for ( auto e : all6Edges )
            {
                if ( !directChild[(int)e] )
                    continue;
                auto neiv = getExistingNeighbor( v, e );
                const auto neis = voxel2seq_[neiv];
                assert( neis );
                [[maybe_unused]] const auto & neid = voxelData_[neis];
                assert( opposite( e ) == neid.parent() );
                assert( neid.side() == side );
                voxelData_[neis].setParent( OutEdge::Invalid );
                orphans_.push_back( neis );
            }
            if ( bestOtherSideParent != OutEdge::Invalid )
            {
                ++growths_;
                vd.setSide( opposite( side ) );
                vd.setParent( bestOtherSideParent );
                assert( checkNotSaturatedPath_( s, opposite( side ) ) );
                active_.set( s );
            }
            else
            {
                vd.setSide( Side::Unknown );
                vd.setParent( OutEdge::Invalid );
            }
        }
    }
}

bool VoxelGraphCut::isGrandparent_( SeqVoxelId s, SeqVoxelId sGrand ) const
{
    while ( s != sGrand )
    {
        auto edgeToParent = voxelData_[s].parent();
        if ( edgeToParent == OutEdge::Invalid )
            return false;
        auto vParent = getExistingNeighbor( seq2voxel_[s], edgeToParent );
        s = voxel2seq_[vParent];
        assert( s );
    }
    return true;
}

bool VoxelGraphCut::checkNotSaturatedPath_( SeqVoxelId s, Side side ) const
{
    assert( side != Side::Unknown );
    for ( ;; )
    {
        const auto & vd = voxelData_[s];
        assert( vd.side() == side );
        auto edgeToParent = vd.parent();
        if ( edgeToParent == OutEdge::Invalid )
            return true;
        const auto v = seq2voxel_[s];
        auto vParent = getExistingNeighbor( v, edgeToParent );
        auto sParent = voxel2seq_[vParent];
        assert( sParent );
        if ( side == Side::Source )
            assert( capacity_[sParent].forOutEdge[(int)opposite( edgeToParent )] > 0 );
        else
            assert( capacity_[s].forOutEdge[(int)edgeToParent] > 0 );
        s = sParent;
    }
}

tl::expected<VoxelBitSet, std::string> segmentVolumeByGraphCut( const SimpleVolume & densityVolume, float k, const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds, ProgressCallback cb )
{
    MR_TIMER

    VoxelGraphCut vgc( densityVolume.dims );
    return vgc.fill( densityVolume, k, sourceSeeds, sinkSeeds, cb );
}

} // namespace MR
