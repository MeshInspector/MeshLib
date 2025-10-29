#include "MRMesh/MRObject.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRMesh/MRObjectSave.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRCube.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRGTest.h"

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
    EXPECT_TRUE( m0->mesh() );
    EXPECT_EQ( m0->mesh()->topology.numValidFaces(), 12 );
    auto m1 = dynamic_cast<const ObjectMesh*>( l->obj->children()[1].get() );
    EXPECT_TRUE( m1 );
    EXPECT_TRUE( m1->mesh() );
    EXPECT_EQ( m1->mesh()->topology.numValidFaces(), 12 );
    // meshes are equal but not shared
    EXPECT_EQ( *m0->mesh(), *m1->mesh() );
    EXPECT_NE( m0->mesh(), m1->mesh() );
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
    EXPECT_TRUE( m0->mesh() );
    EXPECT_EQ( m0->mesh()->topology.numValidFaces(), 12 );
    auto m1 = dynamic_cast<const ObjectMesh*>( l->obj->children()[1].get() );
    EXPECT_TRUE( m1 );
    EXPECT_TRUE( m1->mesh() );
    EXPECT_EQ( m1->mesh()->topology.numValidFaces(), 12 );
    // meshes are shared among two objects
    EXPECT_EQ( m0->mesh(), m1->mesh() );
}

} //namespace MR
