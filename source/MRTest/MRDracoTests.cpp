#include <MRIOExtras/MRDraco.h>
#ifndef MRIOEXTRAS_NO_DRACO
#include <MRMesh/MRColor.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRphmap.h>
#include <gtest/gtest.h>

#include <sstream>

namespace MR
{

TEST( MRIOExtras, DracoMesh )
{
    const auto mesh = makeCube();
    VertColors colors;
    HashMap<Vector3f, Color> pointColors;
    for ( auto v : mesh.topology.getValidVerts() )
    {
        colors.push_back( Color( 30 * int( v ), 255 - 30 * int( v ), 100 ) );
        pointColors[mesh.points[v]] = colors.back();
    }

    std::stringstream ss;
    ASSERT_TRUE( MeshSave::toDrc( mesh, ss, SaveSettings{ .colors = &colors } ) );

    VertColors loadedColors;
    auto loaded = MeshLoad::fromDrc( ss, { .colors = &loadedColors } );
    ASSERT_TRUE( loaded ) << loaded.error();
    EXPECT_EQ( loaded->topology.numValidVerts(), 8 );
    EXPECT_EQ( loaded->topology.numValidFaces(), 12 );
    EXPECT_TRUE( loaded->topology.isClosed() );
    EXPECT_FLOAT_EQ( float( loaded->volume() ), 1.0f );
    ASSERT_EQ( loadedColors.size(), loaded->points.size() );
    for ( auto v : loaded->topology.getValidVerts() )
    {
        auto it = pointColors.find( loaded->points[v] );
        ASSERT_NE( it, pointColors.end() );
        EXPECT_EQ( loadedColors[v], it->second );
    }

    // lossy compression
    std::stringstream ssq;
    DracoSaveOptions options;
    options.positionQuantizationBits = 8;
    ASSERT_TRUE( MeshSave::toDrc( mesh, ssq, options ) );
    auto loadedq = MeshLoad::fromDrc( ssq );
    ASSERT_TRUE( loadedq ) << loadedq.error();
    EXPECT_EQ( loadedq->topology.numValidFaces(), 12 );
    EXPECT_NEAR( loadedq->volume(), 1.0, 0.05 );
}

TEST( MRIOExtras, DracoPoints )
{
    PointCloud cloud;
    VertColors colors;
    HashMap<Vector3f, std::pair<Vector3f, Color>> attrs;
    for ( int i = 0; i < 100; ++i )
    {
        const Vector3f p( float( i ), float( i * i % 17 ), -0.5f * i );
        cloud.points.push_back( p );
        cloud.normals.push_back( Vector3f( 0, float( i % 2 ), float( 1 - i % 2 ) ) );
        colors.push_back( Color( i, 2 * i, 255 - i ) );
        attrs[p] = { cloud.normals.back(), colors.back() };
    }
    cloud.validPoints.resize( cloud.points.size(), true );

    std::stringstream ss;
    ASSERT_TRUE( PointsSave::toDrc( cloud, ss, SaveSettings{ .colors = &colors } ) );

    VertColors loadedColors;
    auto loaded = PointsLoad::fromDrc( ss, { .colors = &loadedColors } );
    ASSERT_TRUE( loaded ) << loaded.error();
    ASSERT_EQ( loaded->points.size(), 100 );
    ASSERT_EQ( loaded->normals.size(), 100 );
    ASSERT_EQ( loadedColors.size(), 100 );
    for ( auto v : loaded->validPoints )
    {
        auto it = attrs.find( loaded->points[v] );
        ASSERT_NE( it, attrs.end() );
        EXPECT_EQ( loaded->normals[v], it->second.first );
        EXPECT_EQ( loadedColors[v], it->second.second );
    }

    // a point cloud is opened as a mesh without triangles
    ss.clear();
    ss.seekg( 0 );
    auto asMesh = MeshLoad::fromDrc( ss );
    ASSERT_TRUE( asMesh ) << asMesh.error();
    EXPECT_EQ( asMesh->points.size(), 100 );
    EXPECT_EQ( asMesh->topology.numValidFaces(), 0 );
}

} // namespace MR
#endif
