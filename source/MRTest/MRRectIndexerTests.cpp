#include <MRMesh/MRRectIndexer.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRVector2.h>
#include "MRGTest.h"

namespace MR
{

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
