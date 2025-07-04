#pragma once
#include "MRVoxelsFwd.h"

#include "openvdb/tree/TreeIterator.h"
#include "openvdb/tree/Tree.h"
#include "openvdb/tree/ValueAccessor.h"
#include "MRMesh/MRProgressCallback.h"

#if defined(MR_PARSING_FOR_ANY_BINDINGS) || defined(MR_COMPILING_ANY_BINDINGS)
// We call functions from those files below. Without those includes, I get undefined references in the bindings, when importing them.
// Maybe we should include those unconditionally, I'm not entirely sure.
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/SignedFloodFill.h>
#endif

#include <thread>

namespace MR
{

/**
 * @brief Class to use in tbb::parallel_reduce for openvdb::tree transformation
 * @details similar to openvdb::RangeProcessor
 * @tparam TreeT tree type
 * @tparam Transformer functor to transform tree
 */
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
            openvdb::math::CoordBBox bbox = i->getNodeBoundingBox();
            if ( !mBBox.empty() )
                bbox.intersect( mBBox );

            if ( !bbox.empty() )
            {
                for ( auto it = bbox.begin(); it != bbox.end(); ++it )
                {
                    mXform( mInAcc, mOutAcc, *it );
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
                for ( auto it = bbox.begin(); it != bbox.end(); ++it )
                {
                    mXform( mInAcc, mOutAcc, *it );
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

/// functor for shifting voxels
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

MRVOXELS_API void translateToZero( openvdb::FloatGrid & grid );

/// This class holds single progress counter for all parallel_reduce threads
/// but report progress only in main thread
class RangeProgress
{
public:
    // Mode of parallel_reduce
    enum class Mode
    {
        Leaves,
        Tiles
    };
    RangeProgress( ProgressCallback cb, size_t size, Mode mode ) :
        size_{ size },
        cb_{ cb },
        mode_{ mode }
    {
        progressThreadId_ = std::this_thread::get_id();
    }
    // if leaves mode add `l` to counter,
    // if tiles mode - add `t` to counter
    void add( size_t l, size_t t )
    {
        if ( mode_ == Mode::Leaves )
            counter_.fetch_add( l );
        else
            counter_.fetch_add( t );
    }
    // retorts progress if called from main thread, otherwise do nothing
    bool reportProgress() const
    {
        if ( !cb_ || progressThreadId_ != std::this_thread::get_id() )
            return true;
        return cb_( float( counter_.load() ) / float( size_ ) );
    }
private:
    std::atomic<size_t> counter_{ 0 };
    size_t size_{ 0 };
    ProgressCallback cb_;
    std::thread::id progressThreadId_;
    Mode mode_;
};

/**
 * @brief Class to use in tbb::parallel_reduce for tree operations that do not require an output tree
 * @tparam TreeT tree type
 * @tparam Proc functor for operations on a tree
 */
template <typename TreeT, typename Proc>
class RangeProcessorSingle
{
public:
    using InterruptFunc = std::function<bool( void )>;
    using ProgressHolder = std::shared_ptr<RangeProgress>;

    using ValueT = typename TreeT::ValueType;

    using LeafIterT = typename TreeT::LeafCIter;
    using TileIterT = typename TreeT::ValueAllCIter;
    using LeafRange = typename openvdb::tree::IteratorRange<LeafIterT>;
    using TileRange = typename openvdb::tree::IteratorRange<TileIterT>;

    using TreeAccessor = typename openvdb::tree::ValueAccessor<const TreeT>;

    RangeProcessorSingle( const openvdb::math::CoordBBox& b, const TreeT& inT, const Proc& proc ) :
        mProc( proc ), mBBox( b ), mInTree( inT ), mInAcc( mInTree )
    {}

    /// Splitting constructor: don't copy the original processor's output tree
    RangeProcessorSingle( RangeProcessorSingle& other, tbb::split ) :
        mProc( other.mProc ),
        mBBox( other.mBBox ),
        mInTree( other.mInTree ),
        mInAcc( mInTree ),
        mInterrupt( other.mInterrupt ),
        mCanceled{ other.mCanceled },
        mProgress( other.mProgress )
    {}

    void setInterrupt( const InterruptFunc& f ) { mInterrupt = f; }
    void setProgressHolder( ProgressHolder progressHolder ) { mProgress = progressHolder; }

    /// Transform each leaf node in the given range.
    void operator()( const LeafRange& rCRef )
    {
        LeafRange r = rCRef;
        leafCount = 0;
        size_t leafCountLast = 0;
        for ( ; r; ++r )
        {
            if ( interrupt() ) break;
            if ( !( leafCount & 0x400 ) )
            {
                if ( !setProgress( leafCount - leafCountLast, tileCount ) )
                    break;
                else
                    leafCountLast = leafCount;
            }

            LeafIterT i = r.iterator();
            openvdb::math::CoordBBox bbox = i->getNodeBoundingBox();
            if ( !mBBox.empty() )
                bbox.intersect( mBBox );

            if ( !bbox.empty() )
            {
                mProc.action( i, mInAcc, bbox );
                ++leafCount;
            }
        }
        setProgress( leafCount - leafCountLast, tileCount );
    }

    /// Transform each non-background tile in the given range.
    void operator()( const TileRange& rCRef )
    {
        auto r = rCRef;
        tileCount = 0;
        size_t tileCountLast = 0;
        for ( ; r; ++r )
        {
            if ( interrupt() ) break;
            if ( !( tileCount & 0x400 ) )
            {
                if ( !setProgress( leafCount, tileCount - tileCountLast ) )
                    break;
                else
                    tileCountLast = tileCount;
            }

            TileIterT i = r.iterator();
            // Skip voxels and background tiles.
            if ( !i.isTileValue() ) continue;
            if ( !i.isValueOn() ) continue;

            openvdb::math::CoordBBox bbox;
            i.getBoundingBox( bbox );
            if ( !mBBox.empty() )
                bbox.intersect( mBBox );

            if ( !bbox.empty() )
            {
                mProc.action( i, mInAcc, bbox );
                ++tileCount;
            }
        }
        setProgress( leafCount, tileCount - tileCountLast );
    }

    /// Merge another processor's output tree into this processor's tree.
    void join( RangeProcessorSingle& other )
    {
        if ( interrupt() )
            return;
        mProc.join( other.mProc );
    }

    Proc mProc;
private:
    bool interrupt() const
    {
        return mCanceled || ( mInterrupt && mInterrupt() );
    }
    bool setProgress( size_t l, size_t t )
    {
        if ( !mProgress )
            return true;
        mProgress->add( l, t );
        if ( !mProgress->reportProgress() )
            mCanceled = true;
        return !mCanceled;
    }

    openvdb::math::CoordBBox mBBox;
    const TreeT& mInTree;
    TreeAccessor mInAcc;
    InterruptFunc mInterrupt;
    bool mCanceled{ false };
    std::shared_ptr<RangeProgress> mProgress;

    size_t leafCount = 0;
    size_t tileCount = 0;
};


struct RangeSize
{
    size_t leaf = 0;
    size_t tile = 0;
};

/// @brief functor to calculate tile and leaf valid nodes count
/// @details valid node - the node where the action is performed.
/// it is necessary to calculate the progress in real action
template<typename TreeT>
class RangeCounter
{
public:
    using ValueT = typename TreeT::ValueType;
    using TreeAccessor = openvdb::tree::ValueAccessor<const TreeT>;
    using LeafIterT = typename TreeT::LeafCIter;
    using TileIterT = typename TreeT::ValueAllCIter;

    RangeCounter()
    {}

    RangeCounter( const RangeCounter& )
    {}

    void action( const LeafIterT&, const TreeAccessor&, const openvdb::math::CoordBBox& )
    {
        ++size.leaf;
    }

    void action( const TileIterT&, const TreeAccessor&, const openvdb::math::CoordBBox& )
    {
        ++size.tile;
    }

    void join( const RangeCounter& other )
    {
        size.leaf += other.size.leaf;
        size.tile += other.size.tile;
    }

    RangeSize size;
};

template <typename GridT>
RangeSize calculateRangeSize( const GridT& grid )
{
    using TreeT = typename GridT::TreeType;
    using ProcessC = RangeCounter<TreeT>;
    ProcessC proc;
    using RangeProcessC = RangeProcessorSingle<TreeT, ProcessC>;
    RangeProcessC calcCount( grid.evalActiveVoxelBoundingBox(), grid.tree(), proc );

    typename RangeProcessC::TileIterT tileIter = grid.tree().cbeginValueAll();
    tileIter.setMaxDepth( tileIter.getLeafDepth() - 1 ); // skip leaf nodes
    typename RangeProcessC::TileRange tileRange( tileIter );
    tbb::parallel_reduce( tileRange, calcCount );
    typename RangeProcessC::LeafRange leafRange( grid.tree().cbeginLeaf() );
    tbb::parallel_reduce( leafRange, calcCount );

    return calcCount.mProc.size;
}

}
