#include "MRMeshEigen.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRGTest.h"

namespace MR
{

MeshTopology topologyFromEigen( const Eigen::MatrixXi & F )
{
    MR_TIMER
    assert( F.cols() == 3 );
    int numTris = (int)F.rows();
    Triangulation t;
    t.reserve( numTris );

    for ( int r = 0; r < numTris; ++r )
        t.push_back( { VertId( F(r, 0) ), VertId( F(r, 1) ), VertId( F(r, 2) ) } );

    return MeshBuilder::fromTriangles( t );
}

Mesh meshFromEigen( const Eigen::MatrixXd & V, const Eigen::MatrixXi & F )
{
    MR_TIMER
    assert( V.cols() == 3 );

    Mesh res;
    res.topology = topologyFromEigen( F );

    int numPoints = (int)V.rows();
    res.points.reserve( numPoints );
    for ( int r = 0; r < numPoints; ++r )
        res.points.emplace_back( (float)V(r, 0), (float)V(r, 1), (float)V(r, 2) );

    return res;
}

void pointsFromEigen( const Eigen::MatrixXd & V, const VertBitSet & selection, VertCoords & points )
{
    MR_TIMER
    for ( auto v : selection )
        points[v] = Vector3f{ (float)V( (int)v, 0), (float)V( (int)v, 1), (float)V( (int)v, 2) };
}

void topologyToEigen( const MeshTopology & topology, Eigen::MatrixXi & F )
{
    MR_TIMER
    int numFaces = topology.numValidFaces();
    F.resize( numFaces, 3 );

    int r = 0;
    for ( const auto & e : topology.edgePerFace() )
    {
        if ( !e.valid() )
            continue;
        VertId a, b, c;
        topology.getLeftTriVerts( e, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );
        F( r, 0 ) = a;
        F( r, 1 ) = b;
        F( r, 2 ) = c;
        ++r;
    }
}

void meshToEigen( const Mesh & mesh, Eigen::MatrixXd & V, Eigen::MatrixXi & F )
{
    MR_TIMER
    topologyToEigen( mesh.topology, F );

    const VertId lastValidPoint = mesh.topology.lastValidVert();
    V.resize( lastValidPoint + 1, 3 );

    for ( int i{ 0 }; i <= lastValidPoint; ++i )
    {
        auto p = mesh.points[ (VertId) i];
        V( i, 0 ) = p.x;
        V( i, 1 ) = p.y;
        V( i, 2 ) = p.z;
    }
}

TEST(MRMesh, Eigen) 
{
    Eigen::MatrixXd V( 3, 3 );
    V( 0, 0 ) = 0; V( 0, 1 ) = 0; V( 0, 2 ) = 0;
    V( 1, 0 ) = 1; V( 1, 1 ) = 0; V( 1, 2 ) = 0;
    V( 2, 0 ) = 0; V( 2, 1 ) = 1; V( 2, 2 ) = 0;

    Eigen::MatrixXi F( 1, 3 );
    F( 0, 0 ) = 0; F( 0, 1 ) = 1; F( 0, 2 ) = 2;

    auto mesh = meshFromEigen( V, F );

    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    meshToEigen( mesh, V1, F1 );

    EXPECT_TRUE( V == V1 );
    EXPECT_TRUE( F == F1 );
}

} //namespace MR
