#include "MRRectIndexer.h"
#include "MRBitSet.h"
#include "MRBitSetParallelFor.h"

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

} //namespace MR
