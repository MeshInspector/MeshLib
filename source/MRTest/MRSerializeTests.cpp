#include "MRMesh/MRObject.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRMesh/MRObjectSave.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRDistanceMeasurementObject.h"
#include "MRMesh/MRPointMeasurementObject.h"
#include "MRMesh/MRCube.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRMeshToPointCloud.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRTelemetry.h"
#include "MRPch/MRJson.h"
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, SerializeObject )
{
    Object o;
    o.setName( "123" );
    UniqueTemporaryFolder f;
    auto mruPath = f / "123.mru";
    auto s = serializeObjectTree( o, mruPath );
    EXPECT_TRUE( s.has_value() );
    auto l = loadSceneFromAnySupportedFormat( mruPath );
    EXPECT_TRUE( l.has_value() );
    EXPECT_TRUE( l->obj.get() );
    EXPECT_TRUE( l->obj->name() == o.name() );
    EXPECT_TRUE( l->warnings.empty() );
}

TEST( MRMesh, SerializeObjectMesh )
{
    auto cubeMesh = std::make_shared<Mesh>( makeCube() );
    EXPECT_EQ( cubeMesh->topology.numValidFaces(), 12 );
    Object o;
    o.setName( "root" );
    auto om = std::make_shared<ObjectMesh>();
    om->setName( "mesh" );
    om->setMesh( cubeMesh );
    o.addChild( om );
    o.addChild( om->clone() );
    UniqueTemporaryFolder f;
    auto mruPath = f / "2meshes.mru";
    auto s = serializeObjectTree( o, mruPath );
    EXPECT_TRUE( s.has_value() );
    auto l = loadSceneFromAnySupportedFormat( mruPath );
    EXPECT_TRUE( l.has_value() );
    EXPECT_TRUE( l->obj );
    EXPECT_TRUE( l->warnings.empty() );
    EXPECT_EQ( l->obj->children().size(), 2 );
    auto m0 = dynamic_cast<const ObjectMesh*>( l->obj->children()[0].get() );
    EXPECT_TRUE( m0 );
    EXPECT_TRUE( m0->meshPtr() );
    EXPECT_EQ( m0->meshPtr()->topology.numValidFaces(), 12 );
    auto m1 = dynamic_cast<const ObjectMesh*>( l->obj->children()[1].get() );
    EXPECT_TRUE( m1 );
    EXPECT_TRUE( m1->meshPtr() );
    EXPECT_EQ( m1->meshPtr()->topology.numValidFaces(), 12 );
    // meshes are equal but not shared
    EXPECT_EQ( *m0->meshPtr(), *m1->meshPtr() );
    EXPECT_NE( m0->meshPtr(), m1->meshPtr() );
}

// writing a scene in .mru file must not report any telemetry about the models saved inside it
TEST( MRMesh, SerializeNoTelemetry )
{
    Object o;
    o.setName( "root" );
    auto om = std::make_shared<ObjectMesh>();
    om->setName( "mesh" );
    om->setMesh( std::make_shared<Mesh>( makeCube() ) );
    o.addChild( om );
    auto cloud = std::make_shared<PointCloud>( meshToPointCloud( *om->meshPtr() ) );
    auto op = std::make_shared<ObjectPoints>();
    op->setName( "points" );
    op->setPointCloud( cloud );
    o.addChild( op );
    auto op1 = std::make_shared<ObjectPoints>();
    op1->setName( "points1" );
    op1->setPointCloud( cloud );
    op1->setSerializeFormat( ".unknown" ); // the only way to reach PointsSave::toAnySupportedFormat from here
    o.addChild( op1 );

    std::vector<std::string> signals;
    boost::signals2::scoped_connection con = TelemetrySignal.connect(
        [&signals]( const std::string& s ) { signals.push_back( s ); } );

    UniqueTemporaryFolder f;
    auto s = serializeObjectTree( o, f / "noTelemetry.mru" );
    EXPECT_TRUE( s.has_value() ) << ( s.has_value() ? "" : s.error() );
    for ( const auto & signal : signals )
        ADD_FAILURE() << "unexpected telemetry during .mru saving: " << signal;

    // in contrast, ordinary saving of the same models is reported
    signals.clear();
    EXPECT_TRUE( MeshSave::toAnySupportedFormat( *om->meshPtr(), f / "cube.ply" ).has_value() );
    EXPECT_EQ( signals, std::vector<std::string>( { "Save *.ply VP TRI", "Save Mesh Log Tris 4" } ) );

    signals.clear();
    EXPECT_TRUE( PointsSave::toAnySupportedFormat( *cloud, f / "cube.xyz" ).has_value() );
    EXPECT_EQ( signals, std::vector<std::string>( { "Save *.xyz VPN", "Save Pnts Log Pnts 4" } ) );
}

// Zendesk #1121: the name is cut to 12 characters, and the cut used to end with a space
TEST( MRMesh, SerializeObjectNameCutOnSpace )
{
    Object o;
    o.setName( "root" );
    auto group = std::make_shared<Object>();
    group->setName( "Planner FTA Teeth" ); // first 12 characters are "Planner FTA "
    o.addChild( group );
    auto om = std::make_shared<ObjectMesh>();
    om->setName( "Tooth_UL1" );
    om->setMesh( std::make_shared<Mesh>( makeCube() ) );
    group->addChild( om );

    UniqueTemporaryFolder f;
    auto mruPath = f / "cutOnSpace.mru";
    auto s = serializeObjectTree( o, mruPath );
    EXPECT_TRUE( s.has_value() ) << ( s.has_value() ? "" : s.error() );
    auto l = loadSceneFromAnySupportedFormat( mruPath );
    EXPECT_TRUE( l.has_value() );
    ASSERT_TRUE( l->obj );
    ASSERT_EQ( l->obj->children().size(), 1 );
    EXPECT_EQ( l->obj->children()[0]->name(), "Planner FTA Teeth" );
    ASSERT_EQ( l->obj->children()[0]->children().size(), 1 );
    auto m = dynamic_cast<const ObjectMesh*>( l->obj->children()[0]->children()[0].get() );
    ASSERT_TRUE( m );
    ASSERT_TRUE( m->meshPtr() );
    EXPECT_EQ( m->meshPtr()->topology.numValidFaces(), 12 );
}

TEST( MRMesh, SerializeSharedObjectMesh )
{
    auto cubeMesh = std::make_shared<Mesh>( makeCube() );
    EXPECT_EQ( cubeMesh->topology.numValidFaces(), 12 );
    Object o;
    o.setName( "root" );
    for ( int i = 0; i < 2; ++i )
    {
        auto om = std::make_shared<ObjectMesh>();
        om->setName( "mesh" + std::to_string( i ) );
        om->setMesh( cubeMesh );
        o.addChild( om );
    }
    UniqueTemporaryFolder f;
    auto mruPath = f / "2sharedMeshes.mru";
    auto s = serializeObjectTree( o, mruPath );
    EXPECT_TRUE( s.has_value() );
    auto l = loadSceneFromAnySupportedFormat( mruPath );
    EXPECT_TRUE( l.has_value() );
    EXPECT_TRUE( l->obj );
    EXPECT_TRUE( l->warnings.empty() );
    EXPECT_EQ( l->obj->children().size(), 2 );
    auto m0 = dynamic_cast<const ObjectMesh*>( l->obj->children()[0].get() );
    EXPECT_TRUE( m0 );
    EXPECT_TRUE( m0->meshPtr() );
    EXPECT_EQ( m0->meshPtr()->topology.numValidFaces(), 12 );
    auto m1 = dynamic_cast<const ObjectMesh*>( l->obj->children()[1].get() );
    EXPECT_TRUE( m1 );
    EXPECT_TRUE( m1->meshPtr() );
    EXPECT_EQ( m1->meshPtr()->topology.numValidFaces(), 12 );
    // meshes are shared among two objects
    EXPECT_EQ( m0->meshPtr(), m1->meshPtr() );
}

// the distance mode (e.g. the distance along X only) must survive saving and loading a scene
TEST( MRMesh, SerializeDistanceMeasurementMode )
{
    using Mode = DistanceMeasurementObject::DistanceMode;
    const std::pair<Mode, float> modesAndDistances[] =
    {
        { Mode::euclidean, 13.f },
        { Mode::euclideanWithSignedDeltasPerAxis, 13.f },
        { Mode::euclideanWithAbsoluteDeltasPerAxis, 13.f },
        { Mode::xAbsolute, 3.f },
        { Mode::yAbsolute, 4.f },
        { Mode::zAbsolute, 12.f },
    };

    Object o;
    o.setName( "root" );
    for ( const auto& [mode, distance] : modesAndDistances )
    {
        auto dm = std::make_shared<DistanceMeasurementObject>();
        dm->setName( "distance" + std::to_string( int( mode ) ) );
        dm->setLocalPoint( { 1.f, 2.f, 3.f } );
        dm->setLocalDelta( { 3.f, -4.f, 12.f } );
        dm->setDistanceMode( mode );
        EXPECT_FLOAT_EQ( dm->computeDistance(), distance );
        o.addChild( dm );
    }

    UniqueTemporaryFolder f;
    auto mruPath = f / "distances.mru";
    auto s = serializeObjectTree( o, mruPath );
    EXPECT_TRUE( s.has_value() ) << ( s.has_value() ? "" : s.error() );
    auto l = loadSceneFromAnySupportedFormat( mruPath );
    ASSERT_TRUE( l.has_value() ) << l.error();
    ASSERT_TRUE( l->obj );
    EXPECT_TRUE( l->warnings.empty() );
    ASSERT_EQ( l->obj->children().size(), std::size( modesAndDistances ) );
    for ( std::size_t i = 0; i < std::size( modesAndDistances ); ++i )
    {
        const auto& [mode, distance] = modesAndDistances[i];
        auto dm = l->obj->children()[i]->asType<DistanceMeasurementObject>();
        ASSERT_TRUE( dm );
        EXPECT_EQ( dm->getDistanceMode(), mode );
        EXPECT_FLOAT_EQ( dm->computeDistance(), distance );
    }
}

// the cap visibility of a point measurement must survive saving and loading a scene
TEST( MRMesh, SerializePointMeasurementCapVisibility )
{
    constexpr auto CapVisibility = PointMeasurementVisualizePropertyType::CapVisibility;
    auto capMask = ViewportMask::all();
    capMask.set( ViewportId( 2 ), false );

    Object o;
    o.setName( "root" );
    auto pm = std::make_shared<PointMeasurementObject>();
    pm->setName( "point" );
    pm->setVisualizePropertyMask( CapVisibility, capMask );
    o.addChild( pm );

    UniqueTemporaryFolder f;
    auto mruPath = f / "point.mru";
    auto s = serializeObjectTree( o, mruPath );
    EXPECT_TRUE( s.has_value() ) << ( s.has_value() ? "" : s.error() );
    auto l = loadSceneFromAnySupportedFormat( mruPath );
    ASSERT_TRUE( l.has_value() ) << l.error();
    ASSERT_TRUE( l->obj );
    EXPECT_TRUE( l->warnings.empty() );
    ASSERT_EQ( l->obj->children().size(), 1 );
    auto loaded = l->obj->children()[0]->asType<PointMeasurementObject>();
    ASSERT_TRUE( loaded );
    EXPECT_EQ( loaded->getVisualizePropertyMask( CapVisibility ), capMask );

    // scenes saved without the key keep the default
    Json::Value root;
    auto futures = pm->serializeRecursive( f, root, 0 );
    ASSERT_TRUE( futures.has_value() ) << futures.error();
    EXPECT_TRUE( futures->empty() );
    ASSERT_TRUE( root.isMember( "CapVisibility" ) );
    root.removeMember( "CapVisibility" );
    PointMeasurementObject old;
    auto d = old.deserializeRecursive( f, root );
    ASSERT_TRUE( d.has_value() ) << d.error();
    EXPECT_EQ( old.getVisualizePropertyMask( CapVisibility ), ViewportMask::all() );
}

// copying all visualize masks (e.g. to a newly added viewport) must include the cap visibility
TEST( MRMesh, PointMeasurementSetAllVisualizeProperties )
{
    constexpr auto CapVisibility = PointMeasurementVisualizePropertyType::CapVisibility;
    auto capMask = ViewportMask::all();
    capMask.set( ViewportId( 2 ), false );

    PointMeasurementObject src;
    src.setVisualizePropertyMask( CapVisibility, capMask );
    const auto props = src.getAllVisualizeProperties();

    PointMeasurementObject dst;
    dst.setAllVisualizeProperties( props );
    EXPECT_EQ( dst.getVisualizePropertyMask( CapVisibility ), capMask );
    EXPECT_EQ( dst.getAllVisualizeProperties(), props );
}

} //namespace MR
