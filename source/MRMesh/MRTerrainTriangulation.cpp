#include "MRTerrainTriangulation.h"
#include "MRMatrix3.h"
#include "MRMatrix2.h"
#include "MRPch/MRTBB.h"
#include "MRMeshCollidePrecise.h"
#include "MRTimer.h"
#include "MRComputeBoundingBox.h"
#include "MRParallelFor.h"
#include "MRPrecisePredicates2.h"
#include "MRId.h"
#include "MR2to3.h"

namespace MR
{

// Based on
// https://www.researchgate.net/publication/221590183_Primitives_for_the_Manipulation_of_General_Subdivisions_and_the_Computation_of_Voronoi_Diagrams
namespace DivideConquerTriangulation
{

class Triangulator
{
struct OrderedVertTag{};
using OVertId = Id<OrderedVertTag>;

public:
    Triangulator( Vector<Vector2i, VertId>&& points, ProgressCallback cb )
    {
        pts_ = std::move( points );
        vertOrder_.resize( pts_.size() );
        
        std::iota( vertOrder_.vec_.begin(), vertOrder_.vec_.end(), VertId( 0 ) );
        if ( !reportProgress( cb, 0.1f ) )
        {
            canceled_ = true;
            return;
        }

        tbb::parallel_sort( vertOrder_.vec_.begin(), vertOrder_.vec_.end(), [&] ( VertId l, VertId r )
        {
            //return std::tuple( pts_[l].x, pts_[l].y, r ) < std::tuple( pts_[r].x, pts_[r].y, l );
            return smaller( { .id = l,.pt = pts_[l].x }, { .id = r,.pt = pts_[r].x } ); // use SoS based predicate
        } );

        //for ( int i = 0; i + 1 < vertOrder_.size(); ++i )
        //{
        //    if ( pts_[vertOrder_.vec_[i]] == pts_[vertOrder_.vec_[i + 1]] )
        //        spdlog::debug( "Same VertId {} == {}", vertOrder_.vec_[i].get(), vertOrder_.vec_[i + 1].get() );
        //}

        //vertOrder_.vec_.erase( std::unique( vertOrder_.vec_.begin(), vertOrder_.vec_.end(), [&] ( VertId l, VertId r )
        //{
        //    return pts_[l] == pts_[r];
        //} ), vertOrder_.vec_.end() );

        if ( !reportProgress( cb, 0.2f ) )
        {
            canceled_ = true;
            return;
        }

        cb_ = subprogress( cb, 0.2f, 1.0f );

        tp_.vertResize( pts_.size() );
    }
    bool isCanceled() const
    {
        return canceled_;
    }
    MeshTopology run()
    {
        if ( canceled_ )
            return {};
        seqDelaunay_( OVertId( 0 ), OVertId( vertOrder_.size() ) );
        //recDelaunay_( OVertId( 0 ), OVertId( vertOrder_.size() ) );
        return std::move( tp_ );
    }
private:
    MeshTopology tp_;
    Vector<Vector2i, VertId> pts_;
    Vector<VertId, OVertId> vertOrder_;
    EdgeId basel_;
    ProgressCallback cb_;
    bool canceled_{ false };

    bool inCircle_( VertId aid, VertId bid, VertId cid, VertId did ) const
    {
        if ( aid == did || bid == did )
            return false; // could be a case in this algorithm

        PreciseVertCoords2 pvc[4];
        pvc[0].id = aid;
        pvc[1].id = bid;
        pvc[2].id = cid;
        pvc[3].id = did;
        for ( int i = 0; i < 4; ++i )
            pvc[i].pt = pts_[pvc[i].id];
        // use SoS based predicate
        return inCircle( pvc );
    }

    bool ccw_( VertId aid, VertId bid, VertId cid ) const
    {
        assert( aid != bid );
        assert( aid != cid );
        assert( bid != cid );

        PreciseVertCoords2 pvc[3];
        pvc[0].id = aid;
        pvc[1].id = bid;
        pvc[2].id = cid;
        for ( int i = 0; i < 3; ++i )
            pvc[i].pt = pts_[pvc[i].id];
        // use SoS based predicate
        return ccw( pvc );
    }

    bool leftOf_( VertId xId, EdgeId e ) const { return ccw_( xId, tp_.org( e ), tp_.dest( e ) ); }
    bool rightOf_( VertId xId, EdgeId e ) const { return ccw_( xId, tp_.dest( e ), tp_.org( e ) ); }
    bool valid_( EdgeId e ) const { return ccw_( tp_.dest( e ), tp_.dest( basel_ ), tp_.org( basel_ ) ); }

    EdgeId connect_( EdgeId a, EdgeId b )
    {
        auto e = tp_.makeEdge();
        tp_.splice( tp_.prev( a.sym() ), e );
        tp_.splice( b, e.sym() );
        return e;
    }

    void deleteEdge_( EdgeId e )
    {
        if ( tp_.left( e ) )
            tp_.setLeft( e, {} );
        if ( tp_.left( e.sym() ) )
            tp_.setLeft( e.sym(), {} );
        tp_.splice( tp_.prev( e ), e );
        tp_.splice( tp_.prev( e.sym() ), e.sym() );
    }

    struct OutEdges
    {
        // has hole to the right
        EdgeId leftMost;
        // has hole to the left
        EdgeId rightMost;
    };

    OutEdges leafDelaunay_( OVertId begin, OVertId end )
    {
        auto size = end - begin;
        assert( size == 2 || size == 3 );

        VertId v0 = vertOrder_[begin];
        VertId v1 = vertOrder_[begin + 1];
        if ( size == 2 )
        {
            auto ne = tp_.makeEdge();
            tp_.setOrg( ne, v0 );
            tp_.setOrg( ne.sym(), v1 );
            return { ne,ne.sym() };
        }
        VertId v2 = vertOrder_[begin + 2];
        EdgeId ne0, ne1;
        {
            ne0 = tp_.makeEdge();
            ne1 = tp_.makeEdge();
            tp_.setOrg( ne0, v0 );
            tp_.setOrg( ne1, v1 );
            tp_.setOrg( ne1.sym(), v2 );
            tp_.splice( ne1, ne0.sym() );
        }
        if ( ccw_( v0, v1, v2 ) )
        {
            connect_( ne1, ne0 );
            tp_.setLeft( ne0, tp_.addFaceId() );
            return { ne0,ne1.sym() };
        }
        else
        {
            assert( ccw_( v0, v2, v1 ) ); // as far as we use SoS, it should always be true
            auto ne2 = connect_( ne1, ne0 );
            tp_.setLeft( ne0.sym(), tp_.addFaceId() );
            return { ne2.sym(),ne2 };
        }
    }

    OutEdges nodeDelaunay_( const OutEdges& leftOut, const OutEdges& rightOut )
    {
        auto [ldo, ldi] = leftOut;
        auto [rdi, rdo] = rightOut;
        for ( ;;)
        {
            if ( leftOf_( tp_.org( rdi ), ldi ) )
                ldi = tp_.prev( ldi.sym() );
            else if ( rightOf_( tp_.org( ldi ), rdi ) )
                rdi = tp_.next( rdi.sym() );
            else
                break;
        }
        basel_ = connect_( rdi.sym(), ldi );

        if ( tp_.org( ldi ) == tp_.org( ldo ) )
            ldo = basel_.sym();
        if ( tp_.org( rdi ) == tp_.org( rdo ) )
            rdo = basel_;

        for ( ;;)
        {
            auto lcand = tp_.next( basel_.sym() );
            if ( valid_( lcand ) )
            {
                while ( inCircle_( tp_.dest( basel_ ), tp_.org( basel_ ), tp_.dest( lcand ), tp_.dest( tp_.next( lcand ) ) ) )
                {
                    auto t = tp_.next( lcand );
                    deleteEdge_( lcand );
                    lcand = t;
                }
            }
            auto rcand = tp_.prev( basel_ );
            if ( valid_( rcand ) )
            {
                while ( inCircle_( tp_.dest( basel_ ), tp_.org( basel_ ), tp_.dest( rcand ), tp_.dest( tp_.prev( rcand ) ) ) )
                {
                    auto t = tp_.prev( rcand );
                    deleteEdge_( rcand );
                    rcand = t;
                }
            }
            if ( !valid_( lcand ) && !valid_( rcand ) )
                break;
            if ( !valid_( lcand ) || ( valid_( rcand ) &&
                inCircle_( tp_.dest( lcand ), tp_.org( lcand ), tp_.org( rcand ), tp_.dest( rcand ) ) ) )
            {
                basel_ = connect_( rcand, basel_.sym() );
            }
            else
            {
                basel_ = connect_( basel_.sym(), lcand.sym() );
            }
            tp_.setLeft( basel_, tp_.addFaceId() );
        }
        assert( tp_.hasEdge( ldo ) );
        assert( tp_.hasEdge( rdo ) );
        return { ldo,rdo };
    }

    OutEdges recDelaunay_( OVertId begin, OVertId end )
    {
        auto size = end - begin;
        if ( size == 2 || size == 3 )
            return leafDelaunay_( begin, end );

        auto left = recDelaunay_( begin, OVertId( ( begin.get() + end.get() ) / 2 ) );
        auto right = recDelaunay_( OVertId( ( begin.get() + end.get() ) / 2 ), end );
        return nodeDelaunay_( left, right );
    }

    OutEdges seqDelaunay_( OVertId begin, OVertId end )
    {
        struct SubTask
        {
            OVertId b, e;
            OutEdges l, r;
            int parentIndex{ INT_MAX };
            bool isLeaf() const
            {
                return e - b < 4;
            }
            bool validNode() const 
            {
                return r.rightMost.valid();
            }
        };
        constexpr int MaxStackSize = 64;
        SubTask subtasks[MaxStackSize];
        int stackSize = 0;

        auto addSubTask = [&] ( OVertId begin, OVertId end, int parent )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = { 
                .b = begin, .e = end,
                .parentIndex = parent
            };
        };

        size_t reporter = 0;
        size_t counter = 0;
        size_t wholeSize = ( end - begin );

        addSubTask( begin, end, INT_MAX );
        while ( stackSize > 0 )
        {
            const auto s = subtasks[stackSize-1];
            if ( s.isLeaf() )
            {
                bool left = s.parentIndex < 0;
                auto indParent = s.parentIndex;
                if ( left )
                    indParent = -( s.parentIndex + 1 );
                auto& res = left ? subtasks[indParent].l : subtasks[indParent].r;
                res = leafDelaunay_( s.b, s.e );
                if ( cb_ )
                    counter += ( s.e - s.b );
                --stackSize;
                continue;
            }
            if ( s.validNode() )
            {
                if ( s.parentIndex == INT_MAX )
                    return nodeDelaunay_( s.l, s.r );

                bool left = s.parentIndex < 0;
                auto indParent = s.parentIndex;
                if ( left )
                    indParent = -( s.parentIndex + 1 );
                auto& res = left ? subtasks[indParent].l : subtasks[indParent].r;
                res = nodeDelaunay_( s.l, s.r );
                if ( cb_ )
                {
                    if ( ( ( reporter++ ) % 512 ) == 0 )
                    {
                        canceled_ = !cb_( float( counter ) / float( wholeSize ) );
                        if ( canceled_ )
                            return {};
                    }
                }
                --stackSize;
                continue;
            }
            auto curInd = stackSize - 1;
            // first add right
            addSubTask( OVertId( ( s.b + s.e ) / 2 ), s.e, curInd );
            // second add left (to process right second)
            addSubTask( s.b, OVertId( ( s.b + s.e ) / 2 ), -curInd - 1 );
        }
        assert( false );
        return {};
    }

    //OutEdges parDelaunay_( VertId begin, VertId end )
    //{
    //    return tbb::parallel_deterministic_reduce( tbb::blocked_range<VertId>( begin, end, 4 ), OutEdges{},
    //        [&] ( const auto& range, OutEdges curr )
    //    {
    //        //assert( range.size() >= 2 && range.size() < 4 );
    //        if ( range.size() > 3 )
    //            curr = recDelaunay_( range.begin(), range.end() );
    //        else
    //            curr = leafDelaunay_( range.begin(), range.end() );
    //        return curr;
    //    },
    //        [this] ( auto a, auto b )
    //    {
    //        return nodeDelaunay_( a, b );
    //    } );
    //}
};

}

Expected<Mesh> terrainTriangulation( std::vector<Vector3f> points, ProgressCallback cb /*= {} */ )
{
    MR_TIMER;

    Mesh resMesh;
    resMesh.points = std::move( points );
    auto box = Box3d( computeBoundingBox( resMesh.points ) );

    auto toInt = getToIntConverter( box );
    Vector<Vector2i, VertId> p2d( resMesh.points.size() );
    ParallelFor( p2d, [&] ( VertId v )
    {
        p2d[v] = to2dim( toInt( resMesh.points[v] ) );
    } );

    if ( cb && !cb( 0.1f ) )
        return unexpectedOperationCanceled();

    DivideConquerTriangulation::Triangulator t( std::move( p2d ), subprogress( cb, 0.1f, 1.0f ) );
    resMesh.topology = std::move( t.run() );
    if ( t.isCanceled() )
        return unexpectedOperationCanceled();

    return resMesh;
}

}