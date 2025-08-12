#include "MRRectIndexer.h"
#include "MRBitSet.h"
#include "MRBitSetParallelFor.h"
#include "MRGTest.h"

namespace MR
{

PixelId RectIndexer::getNeighbor( PixelId v, const Vector2i & pos, OutEdge2 toNei ) const
{
    switch ( toNei )
    {
    default:
        assert( false );
        [[fallthrough]];
    case OutEdge2::Invalid:
        return {};
    case OutEdge2::PlusY:
        return pos.y + 1 < dims_.y ? v + dims_.x : PixelId{};
    case OutEdge2::MinusY:
        return pos.y > 0 ? v - dims_.x : PixelId{};
    case OutEdge2::PlusX:
        return pos.x + 1 < dims_.x ? v + 1 : PixelId{};
    case OutEdge2::MinusX:
        return pos.x > 0 ? v - 1 : PixelId{};
    }
}

void expandPixelMask( PixelBitSet& mask, const RectIndexer& indexer, int expansion )
{
    if ( expansion <= 0 )
    {
        assert( false );
        return;
    }
    PixelBitSet newBitSet( indexer.size() );
    for ( int iter = 0; iter < expansion; ++iter )
    {
        newBitSet.reset();
        BitSetParallelForAll( mask, [&] ( PixelId vId )
        {
            if ( mask.test( vId ) )
                return;
            PixelId edgeVid{};
            bool found = false;
            for ( int step = 0; step < int( OutEdge2::Count ); ++step )
            {
                edgeVid = indexer.getNeighbor( vId, OutEdge2( step ) );
                if ( edgeVid && mask.test( edgeVid ) )
                {
                    found = true;
                    break;
                }
            }
            if ( found )
                newBitSet.set( vId );
        } );
        mask |= newBitSet;
    }
}

void shrinkPixelMask( PixelBitSet& mask, const RectIndexer& indexer, int shrinkage /*= 1 */ )
{
    if ( shrinkage <= 0 )
    {
        assert( false );
        return;
    }
    PixelBitSet newBitSet( indexer.size() );
    for ( int iter = 0; iter < shrinkage; ++iter )
    {
        newBitSet.reset();
        BitSetParallelFor( mask, [&] ( PixelId vId )
        {
            PixelId edgeVid{};
            bool found = false;
            for ( int step = 0; step < int( OutEdge2::Count ); ++step )
            {
                edgeVid = indexer.getNeighbor( vId, OutEdge2( step ) );
                if ( !edgeVid || !mask.test( edgeVid ) )
                {
                    found = true;
                    break;
                }
            }
            if ( found )
                newBitSet.set( vId );
        } );
        mask -= newBitSet;
    }
}

TEST( MRMesh, ExpandShrinkPixels )
{
    RectIndexer indexer( Vector2i::diagonal( 8 ) );
    PixelBitSet mask( indexer.size() );
    mask.set( indexer.toPixelId( { 4, 4 } ) );

    PixelBitSet refMask = mask;
    refMask.set( indexer.toPixelId( { 4, 5 } ) );
    refMask.set( indexer.toPixelId( { 5, 4 } ) );
    refMask.set( indexer.toPixelId( { 4, 3 } ) );
    refMask.set( indexer.toPixelId( { 3, 4 } ) );

    auto storeMask = mask;
    expandPixelMask( mask, indexer );
    EXPECT_TRUE( mask.is_subset_of( refMask ) );
    shrinkPixelMask( mask, indexer );
    EXPECT_TRUE( mask.is_subset_of( storeMask ) );
}

} //namespace MR
