#include "MRVoxelGraphCut.h"
#include "MRVector.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRSimpleVolume.h"
#include "MRHeap.h"
#include "MRVolumeIndexer.h"
#include <spdlog/spdlog.h>
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

class VoxelGraphCut : public VolumeIndexer
{
public:
    VoxelGraphCut( const SimpleVolume & densityVolume, float k );
    VoxelBitSet fill( const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds );

private:
    Vector<VoxelOutEdgeCapacity, VoxelId> capacity_;
    Vector<VoxelData, VoxelId> voxelData_;
    std::deque<VoxelId> active_;
    std::vector<VoxelId> orphans_;
    //statistics:
    int growths_ = 0;
    int augmentations_ = 0;
    int adoptions_ = 0;
    double totalFlow_ = 0;
    //std::ofstream f_{R"(D:\logs\voxelgc.txt)"};

    // return edge capacity:
    //   from v to vnei for Source side and
    //   from vnei to v for Sink side
    float edgeCapacity_( Side side, VoxelId v, OutEdge vOutEdge, VoxelId neiv );
    // constructs initial forest of paths processing vertices in min-edge-capacity-in-path-till-vertex order
    void buildInitialForest_( const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds );
    // process neighborhood of given active voxel
    void processActive_( VoxelId v );
    // augment the path joined at neighbor voxels vSource and vSink
    void augment_( VoxelId vSource, OutEdge vSourceOutEdge, VoxelId vSink );
    // adopt orphans_
    void adopt_();
    // tests whether grand is a grandparent of child
    bool isGrandparent_( VoxelId child, VoxelId grand ) const;
    // checks that there is not saturated path from f to a root
    bool checkNotSaturatedPath_( VoxelId v, Side side ) const;
};

VoxelGraphCut::VoxelGraphCut( const SimpleVolume & densityVolume, float k )
    : VolumeIndexer( densityVolume.dims )
{
    MR_TIMER;

    assert( size_ == densityVolume.data.size() );
    capacity_.resize( size_ );
    voxelData_.resize( size_ );

    // prevent infinite capacities
    constexpr float maxCapacity = FLT_MAX / 10;
    const float maxDelta = log( maxCapacity ) / k;

    auto capacity = [=]( float densityFrom, float densityTo )
    {
        const auto delta = densityTo - densityFrom;
        if ( delta > maxDelta )
            return maxCapacity;
        return std::exp( k * delta );
    };

    tbb::parallel_for( tbb::blocked_range<VoxelId>( VoxelId( 0 ), VoxelId( size_ ) ), [&]( const tbb::blocked_range<VoxelId> & range )
    {
        for ( VoxelId vid = range.begin(); vid != range.end(); ++vid )
        {
            auto & cap = capacity_[vid];
            auto density = densityVolume.data[vid];
            auto pos = toPos( vid );
            if ( pos.x > 0 )
                cap.forOutEdge[ (int)OutEdge::MinusX ] = capacity( density, densityVolume.data[ vid - 1 ] );
            if ( pos.x + 1 < dims_.x )
                cap.forOutEdge[ (int)OutEdge::PlusX ] = capacity( density, densityVolume.data[ vid + 1 ] );
            if ( pos.y > 0 )
                cap.forOutEdge[ (int)OutEdge::MinusY ] = capacity( density, densityVolume.data[ vid - dims_.x ] );
            if ( pos.y + 1 < dims_.y )
                cap.forOutEdge[ (int)OutEdge::PlusY ] = capacity( density, densityVolume.data[ vid + dims_.x ] );
            if ( pos.z > 0 )
                cap.forOutEdge[ (int)OutEdge::MinusZ ] = capacity( density, densityVolume.data[ vid - (int)sizeXY_ ] );
            if ( pos.z + 1 < dims_.z )
                cap.forOutEdge[ (int)OutEdge::PlusZ ] = capacity( density, densityVolume.data[ vid + (int)sizeXY_ ] );
        }
    } );
}

void VoxelGraphCut::buildInitialForest_( const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds )
{
    MR_TIMER;

    assert( size_ == sourceSeeds.size() );
    assert( size_ == sinkSeeds.size() );
    assert( sourceSeeds.count() > 0 );
    assert( sinkSeeds.count() > 0 );
    assert( ( sourceSeeds & sinkSeeds ).count() == 0 );

    Heap<float, VoxelId> minPathCapacity( (int)size_ );

    for ( auto v : sourceSeeds )
    {
        voxelData_[v].setSide( Side::Source );
        minPathCapacity.setLargerValue( v, FLT_MAX );
    }
    for ( auto v : sinkSeeds )
    {
        voxelData_[v].setSide( Side::Sink );
        minPathCapacity.setLargerValue( v, FLT_MAX );
    }

    for (;;)
    {
        auto top = minPathCapacity.setTopValue( -1.0f );
        const VoxelId v{ top.id };
        const auto c{ top.val };
        if ( c <= 0 )
            break;
        const auto side = voxelData_[v].side();
        assert( side != Side::Unknown );
        const auto edgeToParent = voxelData_[v].parent();

        bool neiboursOtherSide = false;
        for ( auto e : all6Edges )
        {
            if ( e == edgeToParent )
                continue;
            auto neiv = getNeighbor( v, e );
            if ( !neiv )
                continue;
            auto currNeiC = minPathCapacity.value( neiv );
            if ( currNeiC < 0 )
            {
                if ( voxelData_[neiv].side() != side )
                {
                    assert( voxelData_[neiv].side() == opposite( side ) );
                    neiboursOtherSide = true;
                }
                continue;
            }
            const float edgeCapacity = edgeCapacity_( side, v, e, neiv );
            const float neic = std::min( c, edgeCapacity );
            if ( neic > currNeiC )
            {
                voxelData_[neiv].setSide( side );
                voxelData_[neiv].setParent( opposite( e ) );
                minPathCapacity.setLargerValue( neiv, neic );
            }
        }
        if ( neiboursOtherSide )
            active_.push_back( v );
    }
}

VoxelBitSet VoxelGraphCut::fill( const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds )
{
    MR_TIMER;

    buildInitialForest_( sourceSeeds, sinkSeeds );

    while ( !active_.empty() )
    {
        auto f = active_.front();
        active_.pop_front();
        processActive_( f );
    }

    VoxelBitSet res( size_ );
    for ( VoxelId v{ 0 }; v < voxelData_.size(); ++v )
        if ( voxelData_[v].side() == Side::Source )
            res.set( v );
    std::stringstream ss;
    ss << "VoxelGraphCut statisitcs:\n"
        "  res.count: " << res.count() << "\n"
        "  source seed count: " << sourceSeeds.count() << "\n"
        "  sink seed count: " << sinkSeeds.count() << "\n"
        "  growths: " << growths_ << "\n"
        "  augmentations: " << augmentations_ << "\n"
        "  adoptions: " << adoptions_ << "\n"
        "  total flow: " << totalFlow_ << std::endl;
    spdlog::info( ss.str() );
    return res;
}

inline float VoxelGraphCut::edgeCapacity_( Side side, VoxelId v, OutEdge vOutEdge, VoxelId neiv )
{
    assert( v && neiv );
    assert( getNeighbor( v, vOutEdge ) == neiv );
    assert( side != Side::Unknown );
    if ( side == Side::Source )
        return capacity_[v].forOutEdge[(int)vOutEdge];
    else
        return capacity_[neiv].forOutEdge[(int)opposite( vOutEdge )];
}

void VoxelGraphCut::processActive_( VoxelId v )
{
    const auto & vd = voxelData_[v];
    const auto side = vd.side();
    if ( vd.side() == Side::Unknown )
        return; // voxel has changed the side since the moment it was put in the queue

    auto pos = toPos( v );
    auto edgeToParent = voxelData_[v].parent();

    for ( auto e : all6Edges )
    {
        if ( e == edgeToParent )
            continue;
        auto neiv = getNeighbor( v, pos, e );
        if ( !neiv )
            continue;
        auto & neid = voxelData_[neiv];
        if ( neid.side() == opposite( side ) )
        {
            if ( side == Side::Source )
                augment_( v, e, neiv );
            else
                augment_( neiv, opposite( e ), v );
            if ( vd.side() != side )
                return; // voxel has changed the side during augmentation
            continue;
        }
        if ( neid.side() == side )
            continue;
        float capacity = edgeCapacity_( side, v, e, neiv );
        if ( capacity > 0 )
        {
            ++growths_;
            neid.setSide( side );
            voxelData_[neiv].setParent( opposite( e ) );
            assert( checkNotSaturatedPath_( neiv, side ) );
            active_.push_back( neiv );
        }
    }
}

void VoxelGraphCut::augment_( VoxelId vSource, OutEdge vSourceOutEdge, VoxelId vSink )
{
    assert( vSource && vSink );
    assert( getNeighbor( vSource, vSourceOutEdge ) == vSink );
    auto & srcD = voxelData_[vSource];
    auto & snkD = voxelData_[vSink];

    for ( int iter = 0;; ++iter )
    {
        assert( srcD.side() == Side::Source );
        assert( snkD.side() == Side::Sink );
        assert( checkNotSaturatedPath_( vSource, Side::Source ) );
        assert( checkNotSaturatedPath_( vSink, Side::Sink ) );

        auto minResidualCapacity = capacity_[ vSource ].forOutEdge[ (int)vSourceOutEdge ];
        assert( minResidualCapacity >= 0 );
        if ( minResidualCapacity == 0 )
            break;
        ++augmentations_;

        for ( auto v = vSource;; )
        {
            assert( voxelData_[v].side() == Side::Source );
            auto edgeToParent = voxelData_[v].parent();
            auto parent = getNeighbor( v, edgeToParent );
            if ( !parent )
                break;
            minResidualCapacity = std::min( minResidualCapacity, capacity_[ parent ].forOutEdge[ (int)opposite( edgeToParent ) ] );
            v = parent;
        }
        for ( auto v = vSink;; )
        {
            assert( voxelData_[v].side() == Side::Sink );
            auto edgeToParent = voxelData_[v].parent();
            auto parent = getNeighbor( v, edgeToParent );
            if ( !parent )
                break;
            minResidualCapacity = std::min( minResidualCapacity, capacity_[ v ].forOutEdge[ (int) edgeToParent ] );
            v = parent;
        }

        assert( minResidualCapacity > 0 );
        capacity_[ vSource ].forOutEdge[ (int)vSourceOutEdge ] -= minResidualCapacity;
        capacity_[ vSink ].forOutEdge[ (int)opposite( vSourceOutEdge ) ] += minResidualCapacity;
        totalFlow_ += minResidualCapacity;
        //f_ << totalFlow_ << '\t' << minResidualCapacity << '\n';

        assert( orphans_.empty() );
        for ( auto v = vSource;; )
        {
            assert( voxelData_[v].side() == Side::Source );
            auto edgeToParent = voxelData_[v].parent();
            auto parent = getNeighbor( v, edgeToParent );
            if ( !parent )
                break;
            capacity_[ v ].forOutEdge[ (int) edgeToParent ] += minResidualCapacity;
            if ( ( capacity_[ parent ].forOutEdge[ (int)opposite( edgeToParent ) ] -= minResidualCapacity ) == 0 )
            {
                voxelData_[v].setParent( OutEdge::Invalid );
                orphans_.push_back( v );
            }
            v = parent;
        }

        for ( auto v = vSink;; )
        {
            assert( voxelData_[v].side() == Side::Sink );
            auto edgeToParent = voxelData_[v].parent();
            auto parent = getNeighbor( v, edgeToParent );
            if ( !parent )
                break;
            capacity_[ parent ].forOutEdge[ (int)opposite( edgeToParent ) ] += minResidualCapacity;
            if ( ( capacity_[ v ].forOutEdge[ (int) edgeToParent ] -= minResidualCapacity ) == 0 )
            {
                voxelData_[v].setParent( OutEdge::Invalid );
                orphans_.push_back( v );
            }
            v = parent;
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
        ++adoptions_;
        const auto v = orphans_.back();
        auto & vd = voxelData_[v];
        orphans_.pop_back();
        const auto side = vd.side();
        assert( side != Side::Unknown );
        assert( vd.parent() == OutEdge::Invalid );
        const auto pos = toPos( v );
        for ( auto e : all6Edges )
        {
            auto neiv = getNeighbor( v, pos, e );
            if ( !neiv )
                continue;
            const auto & neid = voxelData_[neiv];
            if ( neid.side() != side )
                continue;
            float capacity = edgeCapacity_( side, neiv, opposite( e ), v );
            if ( capacity > 0 )
            {
                if ( isGrandparent_( neiv, v ) )
                    active_.push_front( neiv );
                else
                {
                    voxelData_[v].setParent( e );
                    assert( checkNotSaturatedPath_( v, side ) );
                    break;
                }
            }
        }
        if ( voxelData_[v].parent() == OutEdge::Invalid )
        {
            // parent has not been found
            vd.setSide( Side::Unknown );
            OutEdge bestOtherSideParent = OutEdge::Invalid;
            float bestCapacity = 0;
            for ( auto e : all6Edges )
            {
                auto neiv = getNeighbor( v, pos, e );
                if ( !neiv )
                    continue;
                const auto & neid = voxelData_[neiv];
                if ( opposite( e ) == neid.parent() )
                {
                    assert( neid.side() == side );
                    voxelData_[neiv].setParent( OutEdge::Invalid );
                    orphans_.push_back( neiv );
                }
                if ( neid.side() == opposite( side ) )
                {
                    float capacity = edgeCapacity_( side, v, e, neiv );
                    if ( capacity > bestCapacity )
                    {
                        bestOtherSideParent = e;
                        bestCapacity = capacity;
                    }
                }
                if ( bestOtherSideParent != OutEdge::Invalid )
                {
                    vd.setSide( opposite( side ) );
                    vd.setParent( bestOtherSideParent );
                    active_.push_back( v );
                }
            }
        }
    }
}

bool VoxelGraphCut::isGrandparent_( VoxelId v, VoxelId grand ) const
{
    while ( v != grand )
    {
        auto edgeToParent = voxelData_[v].parent();
        auto parent = getNeighbor( v, edgeToParent );
        if ( !parent )
            return false;
        v = parent;
    }
    return true;
}

bool VoxelGraphCut::checkNotSaturatedPath_( VoxelId v, Side side ) const
{
    assert( side != Side::Unknown );
    for ( int iter = 0;; ++iter )
    {
        const auto & vd = voxelData_[v];
        assert( vd.side() == side );
        auto edgeToParent = vd.parent();
        auto parent = getNeighbor( v, edgeToParent );
        if ( !parent )
            return true;
        if ( side == Side::Source )
            assert( capacity_[parent].forOutEdge[(int)opposite( edgeToParent )] > 0 );
        else
            assert( capacity_[v].forOutEdge[(int)edgeToParent] > 0 );
        v = parent;
    }
}

VoxelBitSet segmentVolumeByGraphCut( const SimpleVolume & densityVolume, float k, const VoxelBitSet & sourceSeeds, const VoxelBitSet & sinkSeeds )
{
    MR_TIMER

    VoxelGraphCut vgc( densityVolume, k );
    return vgc.fill( sourceSeeds, sinkSeeds );
}

} // namespace MR
