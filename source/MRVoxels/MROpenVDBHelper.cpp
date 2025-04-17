#include "MRVDBFloatGrid.h"
#include "MROpenVDBHelper.h"
#include "MRMesh/MRTimer.h"

namespace MR
{

void translateToZero( openvdb::FloatGrid & grid )
{
    MR_TIMER;
    using GridT = openvdb::FloatGrid;
    using TreeT = typename GridT::TreeType;
    typename TreeT::Ptr outTreePtr = std::make_shared<TreeT>();
    TreeT& inTree = grid.tree();
    const auto gridClass = grid.getGridClass();
    if (gridClass == openvdb::GRID_LEVEL_SET)
        openvdb::tools::changeLevelSetBackground( *outTreePtr, inTree.background() );

    openvdb::math::CoordBBox bbox = grid.evalActiveVoxelBoundingBox();
    if ( bbox.empty() || bbox.min() == openvdb::math::Coord() )
        return;

    //using RangeProc = RangeProcessor<TreeT, translateValue<TreeT, bbox.min()>>;
    using RangeProc = RangeProcessor<TreeT, ShiftTransformer<TreeT>>;
    ShiftTransformer<TreeT> xform;
    xform.setShift( -bbox.min() );
    RangeProc proc( bbox, inTree, *outTreePtr, xform );

    if ( gridClass != openvdb::GRID_LEVEL_SET )
    {
        // Independently transform the tiles of the input grid.
        // Note: Tiles in level sets can only be background tiles, and they
        // are handled more efficiently with a signed flood fill (see below).
        typename RangeProc::TileIterT tileIter = inTree.cbeginValueAll();
        tileIter.setMaxDepth( tileIter.getLeafDepth() - 1 ); // skip leaf nodes
        typename RangeProc::TileRange tileRange( tileIter );
        tbb::parallel_reduce( tileRange, proc );
    }

    typename RangeProc::LeafRange leafRange( inTree.cbeginLeaf() );
    tbb::parallel_reduce( leafRange, proc );

     if ( gridClass == openvdb::GRID_LEVEL_SET )
     {
         openvdb::tools::pruneLevelSet( *outTreePtr );
         openvdb::tools::signedFloodFill( *outTreePtr );
     }

    grid.setTree( outTreePtr );
}

} //namespace MR
