#include "MRTerrainTriangulation.h"
#include "MRMatrix3.h"
#include "MRMatrix2.h"
#include "MRPch/MRTBB.h"
#include "MRMeshCollidePrecise.h"
#include "MRTimer.h"
#include "MRComputeBoundingBox.h"
#include "MRParallelFor.h"
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
public:
    Triangulator( std::vector<Vector3f>&& points, ProgressCallback cb ):
        cb_{ cb }
    {
        mesh_.points.vec_ = std::move( points );
        mesh_.topology.vertResize( mesh_.points.size() );
    }
    bool isCanceled() const
    {
        return canceled_;
    }
    Mesh run()
    {
        seqDelaunay_( 0_v, VertId( mesh_.points.size() ) );
        return std::move( mesh_ );
    }
private:
    Mesh mesh_;
    EdgeId basel_;
    ProgressCallback cb_;
    bool canceled_{ false };

    bool inCircle_( const Vector3f& af, Vector3f bf, Vector3f cf, Vector3f df ) const
    {
        Vector3d b = Vector3d( bf ) - Vector3d( af );
        Vector3d c = Vector3d( cf ) - Vector3d( af );
        Vector3d d = Vector3d( df ) - Vector3d( af );
        return Matrix3d(
            Vector3d( c.x, c.y, double( c.x ) * c.x + double( c.y ) * c.y ),
            Vector3d( b.x, b.y, double( b.x ) * b.x + double( b.y ) * b.y ),
            Vector3d( d.x, d.y, double( d.x ) * d.x + double( d.y ) * d.y )
        ).det() > 0;
    }

    bool ccw_( const Vector3f& af, Vector3f bf, Vector3f cf ) const
    {
        Vector3d b = Vector3d( bf ) - Vector3d( af );
        Vector3d c = Vector3d( cf ) - Vector3d( af );
        return Matrix2d( to2dim( b ), to2dim( c ) ).det() > 0;
    }

    bool leftOf_( const Vector3f& x, EdgeId e ) const { return ccw_( x, mesh_.orgPnt( e ), mesh_.destPnt( e ) ); }
    bool rightOf_( const Vector3f& x, EdgeId e ) const { return ccw_( x, mesh_.destPnt( e ), mesh_.orgPnt( e ) ); }
    bool valid_( EdgeId e ) const { return ccw_( mesh_.destPnt( e ), mesh_.destPnt( basel_ ), mesh_.orgPnt( basel_ ) ); }

    EdgeId connect_( EdgeId a, EdgeId b )
    {
        auto& tp = mesh_.topology;
        auto e = tp.makeEdge();
        tp.splice( tp.prev( a.sym() ), e );
        tp.splice( b, e.sym() );
        return e;
    }

    void deleteEdge_( EdgeId e )
    {
        auto& tp = mesh_.topology;
        if ( tp.left( e ) )
            tp.setLeft( e, {} );
        if ( tp.left( e.sym() ) )
            tp.setLeft( e.sym(), {} );
        tp.splice( tp.prev( e ), e );
        tp.splice( tp.prev( e.sym() ), e.sym() );
    }

    struct OutEdges
    {
        // has hole to the right
        EdgeId leftMost;
        // has hole to the left
        EdgeId rightMost;
    };

    OutEdges leafDelaunay_( VertId begin, VertId end )
    {
        auto size = end - begin;
        assert( size == 2 || size == 3 );
        auto& m = mesh_;
        auto& tp = m.topology;
        const auto& p = m.points;

        if ( size == 2 )
        {
            auto ne = tp.makeEdge();
            tp.setOrg( ne, VertId( begin ) );
            tp.setOrg( ne.sym(), VertId( begin + 1 ) );
            return { ne,ne.sym() };
        }
        EdgeId ne0, ne1;
        {
            ne0 = tp.makeEdge();
            ne1 = tp.makeEdge();
            tp.setOrg( ne0, VertId( begin ) );
            tp.setOrg( ne1, VertId( begin + 1 ) );
            tp.setOrg( ne1.sym(), VertId( begin + 2 ) );
            tp.splice( ne1, ne0.sym() );
        }
        if ( ccw_( p[begin], p[begin + 1], p[begin + 2] ) )
        {
            connect_( ne1, ne0 );
            tp.setLeft( ne0, tp.addFaceId() );
            return { ne0,ne1.sym() };
        }
        else if ( ccw_( p[begin], p[begin + 2], p[begin + 1] ) )
        {
            auto ne2 = connect_( ne1, ne0 );
            tp.setLeft( ne0.sym(), tp.addFaceId() );
            return { ne2.sym(),ne2 };
        }

        return { ne0,ne1.sym() };
    }

    OutEdges nodeDelaunay_( const OutEdges& leftOut, const OutEdges& rightOut )
    {
        auto& m = mesh_;
        auto& tp = m.topology;
        auto [ldo, ldi] = leftOut;
        auto [rdi, rdo] = rightOut;
        for ( ;;)
        {
            if ( leftOf_( m.orgPnt( rdi ), ldi ) )
                ldi = tp.prev( ldi.sym() );
            else if ( rightOf_( m.orgPnt( ldi ), rdi ) )
                rdi = tp.next( rdi.sym() );
            else
                break;
        }
        basel_ = connect_( rdi.sym(), ldi );

        if ( tp.org( ldi ) == tp.org( ldo ) )
            ldo = basel_.sym();
        if ( tp.org( rdi ) == tp.org( rdo ) )
            rdo = basel_;

        for ( ;;)
        {
            auto lcand = tp.next( basel_.sym() );
            if ( valid_( lcand ) )
            {
                while ( inCircle_( m.destPnt( basel_ ), m.orgPnt( basel_ ), m.destPnt( lcand ), m.destPnt( tp.next( lcand ) ) ) )
                {
                    auto t = tp.next( lcand );
                    if ( t == basel_.sym() )
                        break;
                    if ( lcand.undirected() == ldo.undirected() )
                        break; // precision issues leads to such troubles, for now just break
                    deleteEdge_( lcand );
                    lcand = t;
                }
            }
            auto rcand = tp.prev( basel_ );
            if ( valid_( rcand ) )
            {
                while ( inCircle_( m.destPnt( basel_ ), m.orgPnt( basel_ ), m.destPnt( rcand ), m.destPnt( tp.prev( rcand ) ) ) )
                {
                    auto t = tp.prev( rcand );
                    if ( t == basel_ )
                        break;
                    if ( rcand.undirected() == rdo.undirected() )
                        break; // precision issues leads to such troubles, for now just break
                    deleteEdge_( rcand );
                    rcand = t;
                }
            }
            if ( !valid_( lcand ) && !valid_( rcand ) )
                break;
            if ( !valid_( lcand ) || ( valid_( rcand ) &&
                inCircle_( m.destPnt( lcand ), m.orgPnt( lcand ), m.orgPnt( rcand ), m.destPnt( rcand ) ) ) )
            {
                basel_ = connect_( rcand, basel_.sym() );
            }
            else
            {
                basel_ = connect_( basel_.sym(), lcand.sym() );
            }
            tp.setLeft( basel_, tp.addFaceId() );
        }
        assert( tp.hasEdge( ldo ) );
        assert( tp.hasEdge( rdo ) );
        return { ldo,rdo };
    }

    OutEdges recDelaunay_( VertId begin, VertId end )
    {
        auto size = end - begin;
        if ( size == 2 || size == 3 )
            return leafDelaunay_( begin, end );

        auto left = recDelaunay_( begin, VertId( ( begin.get() + end.get() ) / 2 ) );
        auto right = recDelaunay_( VertId( ( begin.get() + end.get() ) / 2 ), end );
        return nodeDelaunay_( left, right );
    }

    OutEdges seqDelaunay_( VertId begin, VertId end )
    {
        struct SubTask
        {
            VertId b, e;
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

        auto addSubTask = [&] ( VertId begin, VertId end, int parent )
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
            addSubTask( VertId( ( s.b + s.e ) / 2 ), s.e, curInd );
            // second add left (to process right second)
            addSubTask( s.b, VertId( ( s.b + s.e ) / 2 ), -curInd - 1 );
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

    tbb::parallel_sort( points.begin(), points.end(), [] ( const auto& l, const auto& r )->bool
    {
        return l.x < r.x || ( l.x == r.x && l.y < r.y );
    } );

    if ( cb && !cb( 0.1f ) )
        return unexpectedOperationCanceled();

    points.erase( std::unique( points.begin(), points.end(), [] ( const auto& l, const auto& r )->bool
    {
        return l.x == r.x && l.y == r.y;
    } ), points.end() );

    if ( cb && !cb( 0.2f ) )
        return unexpectedOperationCanceled();

    DivideConquerTriangulation::Triangulator t( std::move( points ), subprogress( cb, 0.2f, 1.0f ) );
    auto res = t.run();
    if ( t.isCanceled() )
        return unexpectedOperationCanceled();

    return res;
}

}