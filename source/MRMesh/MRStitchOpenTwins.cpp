#include "MRStitchOpenTwins.h"
#include "MRMesh.h"
#include "MRCloseVertices.h"
#include "MRContoursStitch.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRMeshComponents.h"
#include "MRFinally.h"

namespace MR
{
namespace
{

class TwinStitcher
{
public:
    TwinStitcher( Mesh& mesh, float tolerance, const ProgressCallback& cb ) :
        m_{ mesh },
        t_{ mesh.topology },
        cb_{ cb }
    {
        init_( tolerance );
        p1_.resize( 1 );
        p2_.resize( 1 );
    }

    // runs l2loop_
    Expected<size_t> runLoop();

    // stitches open twins that have only one stitch option
    bool nonAmbiguousPass();

    // stitches open twins that share same vertices (if there are only two in group)
    bool doubleEdgesPass();

    // stitches twin if it has only one stitch option from same component
    bool sameComponentPass();
private:
    Mesh& m_;
    MeshTopology& t_;
    HashMap<VertPair, std::vector<EdgeId>> canonicalMap_;
    UnionFind<FaceId> unionFind_;
    
    EdgePath p1_; // cached vectors
    EdgePath p2_; // cached vectors
    BitSet gbs_; // cached group BitSet

    ProgressCallback cb_;

    size_t curStitches_{ 0 };
    size_t maxStitches_{ 1 }; // 1 - so there is no division by zero on fast cancel mode

    void init_( float tolerance );
    bool tryStitch_( EdgeId e0, EdgeId e1 );

    // for each twin group stitches pair that share same condition only if the pair is unique (there is no 3rd twin in group with same condition)
    bool conditionalPass_( const std::function<void( EdgeId )>& updateStored, const std::function<bool()>& compareStored );

    // calls `doubleEdgesPass` with `nonAmbigousPass` until its done
    bool l1loop_();

    // calls `sameComponentPass` with `l1loop_` until its done
    bool l2loop_();
};

void TwinStitcher::init_( float tolerance )
{
    MR_TIMER;
    MR_FINALLY{ cb_ = subprogress( cb_, 0.5f, 1.0f ); };

    auto closeVertsMapRes = findSmallestCloseVertices( m_, tolerance, subprogress( cb_, 0.0f, 0.2f ) );
    if ( !closeVertsMapRes )
        return;
    const auto& closeVertsMap = *closeVertsMapRes;
    auto twinEdges = findTwinEdgePairs( t_, closeVertsMap );
    UndirectedEdgeBitSet visitedEdges( t_.undirectedEdgeSize() );

    int maxGroupSize = 0;

    auto addE = [&] ( EdgeId e )
    {
        if ( visitedEdges.test_set( e.undirected() ) )
            return;
        auto l = t_.left( e );
        auto r = t_.right( e );
        if ( l.valid() == r.valid() )
            return; // only allow bd edges with one valid face
        auto o = t_.org( e );
        auto d = t_.dest( e );
        auto v0m = closeVertsMap[o];
        auto v1m = closeVertsMap[d];
        if ( v1m == v0m )
            return; // skip degenerated
        if ( v1m < v0m )
        {
            std::swap( v1m, v0m );
            e = e.sym();
        }
        ++maxStitches_; // add for each twin edge in the map
        auto& group = canonicalMap_[std::make_pair( v0m, v1m )];
        group.push_back( e );
        if ( group.size() > maxGroupSize )
            maxGroupSize = int( group.size() );
    };

    auto sb = subprogress( cb_, 0.2f, 0.5f );
    for ( size_t i = 0; i < twinEdges.size(); ++i )
    {
        auto [e0, e1] = twinEdges[i];
        addE( e0 );
        addE( e1 );
        if ( i % 64 == 0 && !reportProgress( sb, float( i ) / float( twinEdges.size() ) ) )
            return;
    }
    gbs_.resize( maxGroupSize );
    unionFind_ = UnionFind<FaceId>( MeshComponents::getUnionFindStructureFaces( m_, MeshComponents::FaceIncidence::PerVertex ) );
    curStitches_ = 0;
    maxStitches_ = maxStitches_ / 2; // each two edge represent 1 stitch
}

bool TwinStitcher::tryStitch_( EdgeId e1, EdgeId e2 )
{
    auto l1 = t_.left( e1 );
    auto r1 = t_.right( e1 );
    auto l2 = t_.left( e2 );
    auto r2 = t_.right( e2 );
    bool e1Rb = !l1 && r1;
    bool e1Lb = l1 && !r1;
    bool e2Rb = !l2 && r2;
    bool e2Lb = l2 && !r2;
    bool stitch = false;
    if ( e1Rb && e2Lb )
    {
        p1_.front() = e1;
        p2_.front() = e2;
        unionFind_.unite( r1, l2 );
        stitch = true;
    }
    else if ( e2Rb && e1Lb )
    {
        p1_.front() = e2;
        p2_.front() = e1;
        unionFind_.unite( l1, r2 );
        stitch = true;
    }
    if ( stitch )
    {
        stitchContours( t_, p1_, p2_ );
        ++curStitches_;
    }
    return stitch;
}

bool TwinStitcher::conditionalPass_( const std::function<void( EdgeId )>& updateStored, const std::function<bool()>& compareStored )
{
    if ( !reportProgress( cb_, float( curStitches_ ) / float( maxStitches_ ) ) )
        return false;
    auto& visited = gbs_;
    size_t counter = 0;
    for ( auto it = canonicalMap_.begin(); it != canonicalMap_.end(); )
    {
        auto& tws = it->second;

        visited.reset();
        visited.resize( tws.size() );
        while ( !visited.all() )
        {
            int numDuplicates = 0;
            int e0 = -1, e1 = -1;
            updateStored( {} );
            for ( int i = 0; i < tws.size(); ++i )
            {
                if ( visited.test( i ) )
                    continue;
                auto e = tws[i];
                updateStored( e );
                if ( !compareStored() )
                    continue;
                visited.set( i );
                if ( e0 == -1 )
                    e0 = i;
                else if ( e1 == -1 )
                    e1 = i;
                ++numDuplicates;
            }
            if ( numDuplicates == 2 )
            {
                if ( tryStitch_( tws[e0], tws[e1] ) )
                {
                    tws[e0] = EdgeId( -1 ); // invalidate
                    tws[e1] = EdgeId( -1 ); // invalidate
                }
            }
        }
        // remove stitched edges
        tws.erase( std::remove_if( tws.begin(), tws.end(), [] ( auto t )
        {
            return !t;
        } ), tws.end() );
        if ( tws.size() <= 1 )
            it = canonicalMap_.erase( it );
        else
            it++;

        if ( counter % 128 == 0 && !reportProgress( cb_, float( curStitches_ ) / float( maxStitches_ ) ) )
            return false;
    }
    if ( !reportProgress( cb_, float( curStitches_ ) / float( maxStitches_ ) ) )
        return false;
    return true;
}

bool TwinStitcher::nonAmbiguousPass()
{
    MR_TIMER;
    return conditionalPass_( [] ( EdgeId ) {}, [] () { return true; } );
}

bool TwinStitcher::doubleEdgesPass()
{
    MR_TIMER;
    VertPair e0, curE;
    return conditionalPass_( [&] ( EdgeId e )
    {
        if ( !e )
        {
            e0 = curE = {};
            return;
        }
        auto o = t_.org( e );
        auto d = t_.dest( e );
        curE = { o,d };
        if ( !e0.first )
            e0 = curE;
    }, [&] ()
    {
        return e0 == curE;
    } );
}

bool TwinStitcher::sameComponentPass()
{
    MR_TIMER;

    auto getRootId = [&] ( EdgeId e )->FaceId
    {
        if ( auto l = t_.left( e ) )
            return unionFind_.find( l );
        if ( auto r = t_.right( e ) )
            return unionFind_.find( r );
        assert( false );
        return {};
    };

    FaceId e0, curE;
    return conditionalPass_( [&] ( EdgeId e )
    {
        if ( !e )
        {
            e0 = curE = {};
            return;
        }
        curE = getRootId( e );
        if ( !e0 )
            e0 = curE;
    }, [&] ()
    {
        return e0 == curE;
    } );
}

bool TwinStitcher::l1loop_()
{
    if ( !nonAmbiguousPass() )
        return false;
    for ( ;;)
    {
        auto lsc = curStitches_;
        if ( !doubleEdgesPass() )
            return false;
        if ( curStitches_ > lsc && !nonAmbiguousPass() )
            return false;
        if ( curStitches_ == lsc )
            break;
    }
    return true;
}

bool TwinStitcher::l2loop_()
{
    if ( !l1loop_() )
        return false;
    for ( ;;)
    {
        auto lsc = curStitches_;
        if ( !sameComponentPass() )
            return false;
        if ( curStitches_ > lsc && !l1loop_() )
            return false;
        if ( curStitches_ == lsc )
            break;
    }
    return true;
}

Expected<size_t> TwinStitcher::runLoop()
{
    if ( !l2loop_() )
        return unexpectedOperationCanceled();
    return curStitches_;
}

}

Expected<size_t> stitchOpenTwinEdges( Mesh& mesh, float tolerance, const ProgressCallback& cb )
{
    MR_TIMER;
    TwinStitcher ts( mesh, tolerance, cb );
    auto numStitched = ts.runLoop();
    if ( numStitched.has_value() && *numStitched > 0 )
        mesh.invalidateCaches();
    return numStitched;
}

}
