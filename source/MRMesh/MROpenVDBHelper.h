#pragma once
#include "MRPch/MROpenvdb.h"
#include "openvdb/tree/TreeIterator.h"
#include "openvdb/tree/Tree.h"
#include "openvdb/tree/ValueAccessor.h"

namespace MR
{

template <class TreeT, typename Transformer>
class RangeProcessor
{
public:
    using InterruptFunc = std::function<bool( void )>;

    using ValueT = typename TreeT::ValueType;

    using LeafIterT = typename TreeT::LeafCIter;
    using TileIterT = typename TreeT::ValueAllCIter;
    using LeafRange = typename openvdb::tree::IteratorRange<LeafIterT>;
    using TileRange = typename openvdb::tree::IteratorRange<TileIterT>;

    using InTreeAccessor = typename openvdb::tree::ValueAccessor<const TreeT>;
    using OutTreeAccessor = typename openvdb::tree::ValueAccessor<TreeT>;

    RangeProcessor( const openvdb::math::CoordBBox& b, const TreeT& inT, TreeT& outT, const Transformer& xform ) :
        mIsRoot( true ), mXform( xform ), mBBox( b ),
        mInTree( inT ), mOutTree( &outT ), mInAcc( mInTree ), mOutAcc( *mOutTree )
    {}

    RangeProcessor( const openvdb::math::CoordBBox& b, const TreeT& inTree, const Transformer& xform ) :
        mIsRoot( false ), mXform( xform ), mBBox( b ),
        mInTree( inTree ), mOutTree( new TreeT( inTree.background() ) ),
        mInAcc( mInTree ), mOutAcc( *mOutTree )
    {}

    ~RangeProcessor()
    {
        if ( !mIsRoot ) delete mOutTree;
    }

    /// Splitting constructor: don't copy the original processor's output tree
    RangeProcessor( RangeProcessor& other, tbb::split ) :
        mIsRoot( false ),
        mXform( other.mXform ),
        mBBox( other.mBBox ),
        mInTree( other.mInTree ),
        mOutTree( new TreeT( mInTree.background() ) ),
        mInAcc( mInTree ),
        mOutAcc( *mOutTree ),
        mInterrupt( other.mInterrupt )
    {}

    void setInterrupt( const InterruptFunc& f ) { mInterrupt = f; }

    /// Transform each leaf node in the given range.
    void operator()( const LeafRange& rCRef )
    {
        LeafRange r = rCRef;
        for ( ; r; ++r )
        {
            if ( interrupt() ) break;

            LeafIterT i = r.iterator();
            openvdb::math::CoordBBox bbox( i->origin(), i->origin() + openvdb::math::Coord( i->dim() ) );
            if ( !mBBox.empty() )
            {
                // Intersect the leaf node's bounding box with mBBox.
                bbox = openvdb::math::CoordBBox(
                    openvdb::math::Coord::maxComponent( bbox.min(), mBBox.min() ),
                    openvdb::math::Coord::minComponent( bbox.max(), mBBox.max() ) );
            }
            if ( !bbox.empty() )
            {
                openvdb::math::Coord bboxMin = bbox.min();
                openvdb::math::Coord bboxMax = bbox.max();
                for ( int ix = bboxMin.x(); ix < bboxMax.x(); ++ix )
                for ( int iy = bboxMin.y(); iy < bboxMax.y(); ++iy )
                for ( int iz = bboxMin.z(); iz < bboxMax.z(); ++iz )
                {
                    mXform( mInAcc, mOutAcc, { ix, iy, iz } );
                }
            }
        }
    }

    /// Transform each non-background tile in the given range.
    void operator()( const TileRange& rCRef )
    {
        auto r = rCRef;
        for ( ; r; ++r )
        {
            if ( interrupt() ) break;

            TileIterT i = r.iterator();
            // Skip voxels and background tiles.
            if ( !i.isTileValue() ) continue;
            if ( !i.isValueOn() && openvdb::math::isApproxEqual( *i, mOutTree->background() ) ) continue;

            openvdb::math::CoordBBox bbox;
            i.getBoundingBox( bbox );
            if ( !mBBox.empty() )
            {
                // Intersect the tile's bounding box with mBBox.
                bbox = openvdb::math::CoordBBox(
                    openvdb::math::Coord::maxComponent( bbox.min(), mBBox.min() ),
                    openvdb::math::Coord::minComponent( bbox.max(), mBBox.max() ) );
            }
            if ( !bbox.empty() )
            {
                openvdb::math::Coord bboxMin = bbox.min();
                openvdb::math::Coord bboxMax = bbox.max();

                for ( int ix = bboxMin.x(); ix < bboxMax.x(); ++ix )
                for ( int iy = bboxMin.y(); iy < bboxMax.y(); ++iy )
                for ( int iz = bboxMin.z(); iz < bboxMax.z(); ++iz )
                {
                    mXform( mInAcc, mOutAcc, { ix, iy, iz } );
                }
            }
        }
    }

    /// Merge another processor's output tree into this processor's tree.
    void join( RangeProcessor& other )
    {
        if ( !interrupt() ) mOutTree->merge( *other.mOutTree );
    }

private:
    bool interrupt() const { return mInterrupt && mInterrupt(); }

    const bool mIsRoot; // true if mOutTree is the top-level tree
    Transformer mXform;
    openvdb::math::CoordBBox mBBox;
    const TreeT& mInTree;
    TreeT* mOutTree;
    InTreeAccessor mInAcc;
    OutTreeAccessor mOutAcc;
    InterruptFunc mInterrupt;
};


template <typename TreeT>
class ShiftTransformer
{
public:
    using InTreeAccessor = typename openvdb::tree::ValueAccessor<const TreeT>;
    using OutTreeAccessor = typename openvdb::tree::ValueAccessor<TreeT>;
    using ValueT = typename TreeT::ValueType;

    void operator()( const InTreeAccessor& inAcc, OutTreeAccessor& outAcc, openvdb::math::Coord coord )
    {
        ValueT value = ValueT();
        if ( inAcc.probeValue( coord, value ) )
            outAcc.setValue( coord + shift_, value );
    }
    void setShift( const openvdb::math::Coord& shift ) { shift_ = shift; }
private:
    openvdb::math::Coord shift_;
};

template <typename GredT>
void translateToZero( GredT& grid)
{
    using TreeT = typename GredT::TreeType;
    typename TreeT::Ptr outTreePtr = std::make_shared<TreeT>();
    TreeT& inTree = grid.tree();
    openvdb::tools::changeLevelSetBackground( *outTreePtr, inTree.background() );

    openvdb::math::CoordBBox bbox = grid.evalActiveVoxelBoundingBox();
    if ( bbox.empty() || bbox.min() == openvdb::math::Coord() )
        return;

    //using RangeProc = RangeProcessor<TreeT, translateValue<TreeT, bbox.min()>>;
    using RangeProc = RangeProcessor<TreeT, ShiftTransformer<TreeT>>;
    ShiftTransformer<TreeT> xform;
    xform.setShift( -bbox.min() );
    RangeProc proc( bbox, inTree, *outTreePtr, xform );

    const auto gridClass = grid.getGridClass();
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

}
