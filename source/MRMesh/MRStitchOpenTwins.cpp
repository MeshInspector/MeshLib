#include "MRStitchOpenTwins.h"
#include "MRMesh.h"
#include "MRCloseVertices.h"
#include "MRContoursStitch.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRMeshComponents.h"

namespace
{
using namespace MR;

class TwinStitcher
{
public:
    TwinStitcher( Mesh& mesh, float tolerance ) :
        m_{ mesh },
        t_{ mesh.topology }
    {
        init_( tolerance );
        p1_.resize( 1 );
        p2_.resize( 1 );
    }

    // runs existing stitch loops until it can
    size_t runLoop();

    // stitches open twins that have only one sticth option
    size_t nonAmbigousPass();

    // stitches open twins that share same vertices (if there there are only two in group)
    size_t doubleEdgesPass();

    // stitches twin if it has only one stitch option from same component
    size_t sameComponentPass();
private:
    Mesh& m_;
    MeshTopology& t_;
    HashMap<VertPair, std::vector<EdgeId>> canonicalMap_; 
    
    EdgePath p1_; // cached vectors
    EdgePath p2_; // cached vectors
    BitSet gbs_; // cached group BitSet

    void init_( float tolerance );
    bool tryStitch_( EdgeId e0, EdgeId e1 );

    // for each twin group stitches pair that share same condition only if the pair is unique (there is no 3rd twin in group with same condition)
    size_t conditionalPass_( const std::function<void( EdgeId )>& updateStored, const std::function<bool()>& compareStored );

    // calls `doubleEdgesPass` with `nonAmbigousPass` untill its done
    size_t l1loop_();

    // calls `sameComponentPass` with `l1loop_` untill its done
    size_t l2loop_();
};

void TwinStitcher::init_( float tolerance )
{
    MR_TIMER;
    auto closeVertsMap = *findSmallestCloseVertices( m_, tolerance );
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
        auto& group = canonicalMap_[std::make_pair( v0m, v1m )];
        group.push_back( e );
        if ( group.size() > maxGroupSize )
            maxGroupSize = int( group.size() );
    };

    for ( auto [e0, e1] : twinEdges )
    {
        addE( e0 );
        addE( e1 );
    }
    gbs_.resize( maxGroupSize );
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
        stitch = true;
    }
    else if ( e2Rb && e1Lb )
    {
        p1_.front() = e2;
        p2_.front() = e1;
        stitch = true;
    }
    if ( stitch )
        stitchContours( t_, p1_, p2_ );
    return stitch;
}

size_t TwinStitcher::nonAmbigousPass()
{
    MR_TIMER;
    size_t counter = 0;
    for ( auto it = canonicalMap_.begin(); it != canonicalMap_.end(); )
    {
        const auto& tws = it->second;
        if ( tws.size() <= 1 || ( tws.size() == 2 && tryStitch_( tws[0], tws[1] ) ) )
        {
            it = canonicalMap_.erase( it );
            ++counter;
        }
        else
            it++;
    }
    return counter;
}

size_t TwinStitcher::conditionalPass_( const std::function<void( EdgeId )>& updateStored, const std::function<bool()>& compareStored )
{
    size_t counter = 0;
    auto& visited = gbs_;
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
            VertId v0, v1;
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
                    ++counter;
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
    }
    return counter;
}

size_t TwinStitcher::doubleEdgesPass()
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

size_t TwinStitcher::sameComponentPass()
{
    MR_TIMER;
    auto compMapAndNum = MeshComponents::getAllComponentsMap( m_, MeshComponents::FaceIncidence::PerVertex );

    auto getCompId = [&] ( EdgeId e )->RegionId
    {
        if ( auto l = t_.left( e ) )
            return compMapAndNum.first[l];
        if ( auto r = t_.right( e ) )
            return compMapAndNum.first[r];
        assert( false );
        return {};
    };

    RegionId e0, curE;
    return conditionalPass_( [&] ( EdgeId e )
    {
        if ( !e )
        {
            e0 = curE = {};
            return;
        }
        curE = getCompId( e );
        if ( !e0 )
            e0 = curE;
    }, [&] ()
    {
        return e0 == curE;
    } );
}

size_t TwinStitcher::l1loop_()
{
    size_t sumStithces = nonAmbigousPass();
    for ( ;;)
    {
        auto localStitches = doubleEdgesPass();
        if ( localStitches > 0 )
            localStitches += nonAmbigousPass();
        sumStithces += localStitches;
        if ( localStitches == 0 )
            break;
    }
    return sumStithces;
}

size_t TwinStitcher::l2loop_()
{
    size_t sumStithces = l1loop_();
    for ( ;;)
    {
        auto localStitches = sameComponentPass();
        if ( localStitches > 0 )
            localStitches += l1loop_();
        sumStithces += localStitches;
        if ( localStitches == 0 )
            break;
    }
    return sumStithces;
}

size_t TwinStitcher::runLoop()
{
    return l2loop_();
}

}

namespace MR
{

void stitchOpenTwinEdges( Mesh& mesh, float tolerance )
{
    MR_TIMER;
    TwinStitcher ts( mesh, tolerance );
    if ( ts.runLoop() > 0 )
        mesh.invalidateCaches();
}

}
