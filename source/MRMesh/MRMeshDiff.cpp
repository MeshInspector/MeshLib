#include "MRMeshDiff.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRMeshBuilder.h"
#include "MRGTest.h"

namespace MR
{

MeshDiff::MeshDiff( const Mesh & from, const Mesh & to )
{
    MR_TIMER

    toPointsSize_ = to.points.size();
    for ( VertId v{0}; v < toPointsSize_; ++v )
    {
        if ( v >= from.points.size() || from.points[v] != to.points[v] )
            changedPoints_[v] = to.points[v];
    }

    toEdgesSize_ = to.topology.edges_.size();
    for ( EdgeId e{0}; e < toEdgesSize_; ++e )
    {
        if ( e >= from.topology.edges_.size() || from.topology.edges_[e] != to.topology.edges_[e] )
            changedEdges_[e] = to.topology.edges_[e];
    }
}

void MeshDiff::applyAndSwap( Mesh & m )
{
    MR_TIMER

    auto mPointsSize = m.points.size();
    // remember points being deleted from m
    for ( VertId v{toPointsSize_}; v < mPointsSize; ++v )
    {
        changedPoints_[v] = m.points[v];
    }
    m.points.resize( toPointsSize_ );
    // swap common points and delete points for vertices missing in original m (that will be next target)
    for ( auto it = changedPoints_.begin(); it != changedPoints_.end(); )
    {
        auto v = it->first;
        auto & pos = it->second;
        if ( v < toPointsSize_ )
        {
            std::swap( pos, m.points[v] );
            if ( v >= mPointsSize )
            {
                it = changedPoints_.erase( it );
                continue;
            }
        }
        ++it;
    }
    toPointsSize_ = mPointsSize;

    auto mEdgesSize = m.topology.edges_.size();
    // remember topology.edges_ being deleted from m
    for ( EdgeId e{toEdgesSize_}; e < mEdgesSize; ++e )
    {
        changedEdges_[e] = m.topology.edges_[e];
    }
    m.topology.edges_.resize( toEdgesSize_ );
    // swap common topology.edges_ and delete topology.edges_ for vertices missing in original m (that will be next target)
    for ( auto it = changedEdges_.begin(); it != changedEdges_.end(); )
    {
        auto e = it->first;
        auto & pos = it->second;
        if ( e < toEdgesSize_ )
        {
            std::swap( pos, m.topology.edges_[e] );
            if ( e >= mEdgesSize )
            {
                it = changedEdges_.erase( it );
                continue;
            }
        }
        ++it;
    }
    toEdgesSize_ = mEdgesSize;

    m.topology.computeAllFromEdges_();
}

TEST(MRMesh, MeshDiff)
{
    Triangulation t
    { 
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v }
    };
    Mesh mesh0;
    mesh0.topology = MeshBuilder::fromTriangles( t );
    mesh0.points.emplace_back( 0.f, 0.f, 0.f );
    mesh0.points.emplace_back( 1.f, 0.f, 0.f );
    mesh0.points.emplace_back( 1.f, 1.f, 0.f );
    mesh0.points.emplace_back( 0.f, 1.f, 0.f );

    Mesh mesh1 = mesh0;
    mesh1.topology.deleteFace( 1_f );
    mesh1.points.pop_back();

    MeshDiff diff( mesh0, mesh1 );
    EXPECT_EQ( diff.any(), true );
    Mesh m = mesh0;
    EXPECT_EQ( m, mesh0 );
    diff.applyAndSwap( m );
    EXPECT_EQ( diff.any(), true );
    EXPECT_EQ( m, mesh1 );
    diff.applyAndSwap( m );
    EXPECT_EQ( diff.any(), true );
    EXPECT_EQ( m, mesh0 );

    EXPECT_EQ( MeshDiff( m, m ).any(), false );
}

} // namespace MR
