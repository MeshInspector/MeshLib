#include "MRVDBFloatGrid.h"
#include "MROpenVDBHelper.h"
#include "MRMesh/MRHistogram.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRTBB.h"

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

/// @brief class to parallel reduce histogram calculation
template<typename TreeT>
class HistogramCalcProc
{
public:
    using ValueT = typename TreeT::ValueType;
    using TreeAccessor = openvdb::tree::ValueAccessor<const TreeT>;
    using LeafIterT = typename TreeT::LeafCIter;
    using TileIterT = typename TreeT::ValueAllCIter;

    HistogramCalcProc( float min, float max, size_t binsNumber ) :
        hist( min, max, binsNumber )
    {}

    HistogramCalcProc( const HistogramCalcProc& other ) :
        hist( Histogram( other.hist.getMin(), other.hist.getMax(), other.hist.getBins().size() ) )
    {}

    void action( const LeafIterT&, const TreeAccessor& treeAcc, const openvdb::math::CoordBBox& bbox )
    {
        for ( auto it = bbox.begin(); it != bbox.end(); ++it )
        {
            ValueT value = ValueT();
            if ( treeAcc.probeValue( *it, value ) )
                hist.addSample( value );
        }
    }

    void action( const TileIterT& iter, const TreeAccessor&, const openvdb::math::CoordBBox& bbox )
    {
        ValueT value = iter.getValue();
        const size_t count = size_t( bbox.volume() );
        hist.addSample( value, count );
    }

    void join( const HistogramCalcProc& other )
    {
        hist.addHistogram( other.hist );
    }

    Histogram hist;
};

Histogram calculateHistogram( const openvdb::FloatGrid& grid, float min, float max, size_t binsNumber, ProgressCallback cb )
{
    RangeSize size = calculateRangeSize( grid );

    using HistogramCalcProcFT = HistogramCalcProc<openvdb::FloatTree>;
    HistogramCalcProcFT histCalcProc( min, max, binsNumber );
    using HistRangeProcessorOne = RangeProcessorSingle<openvdb::FloatTree, HistogramCalcProcFT>;
    HistRangeProcessorOne calc( grid.evalActiveVoxelBoundingBox(), grid.tree(), histCalcProc );

    if ( size.tile > 0 )
    {
        typename HistRangeProcessorOne::TileIterT tileIterMain = grid.tree().cbeginValueAll();
        tileIterMain.setMaxDepth( tileIterMain.getLeafDepth() - 1 ); // skip leaf nodes
        typename HistRangeProcessorOne::TileRange tileRangeMain( tileIterMain );
        auto sb = size.leaf > 0 ? subprogress( cb, 0.0f, 0.5f ) : cb;
        calc.setProgressHolder( std::make_shared<RangeProgress>( sb, size.tile, RangeProgress::Mode::Tiles ) );
        tbb::parallel_reduce( tileRangeMain, calc );
    }

    if ( size.leaf > 0 )
    {
        typename HistRangeProcessorOne::LeafRange leafRangeMain( grid.tree().cbeginLeaf() );
        auto sb = size.tile > 0 ? subprogress( cb, 0.5f, 1.0f ) : cb;
        calc.setProgressHolder( std::make_shared<RangeProgress>( sb, size.leaf, RangeProgress::Mode::Leaves ) );
        tbb::parallel_reduce( leafRangeMain, calc );
    }

    return calc.mProc.hist;
}

void setActiveBounds( openvdb::FloatGrid& grid, const Box3i& box, ProgressCallback cb )
{
    openvdb::CoordBBox activeVdbBox;
    activeVdbBox.min() = openvdb::Coord( box.min.x, box.min.y, box.min.z );
    activeVdbBox.max() = openvdb::Coord( box.max.x - 1, box.max.y - 1, box.max.z - 1 );

    // create active mask tree
    openvdb::TopologyTree topologyTree;

    reportProgress( cb, 0.25f );

    // update topology tree with new active box
    topologyTree.sparseFill( activeVdbBox, true );

    reportProgress( cb, 0.5f );

    // deactivate all of current grid
    openvdb::tools::foreach( grid.tree().beginValueOn(), [] ( const openvdb::FloatTree::ValueOnIter& iter )
    {
        iter.setActiveState( false );
    }, false ); // looks like this operation is not safe to do in threaded mode

    reportProgress( cb, 0.75f );

    // copy valid topology to our tree part
    grid.tree().topologyUnion( topologyTree );

    reportProgress( cb, 1.0f );
}

} //namespace MR
