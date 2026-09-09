#include "MRMesh/MRObject.h"
#include "MRMesh/MRAffineXf3.h"
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, DataModelRemoveChild )
{
    auto child2 = std::make_shared<Object>();
    Object root;
    {
        EXPECT_EQ( root.children().size(), 0 );

        auto child1 = std::make_shared<Object>();
        EXPECT_TRUE( root.addChild( child1 ) );
        EXPECT_FALSE( root.addChild( child1 ) );
        EXPECT_EQ( &root, child1->parent() );
        EXPECT_EQ( root.children().size(), 1 );


        EXPECT_TRUE( child1->addChild( child2 ) );
        EXPECT_FALSE( child1->addChild( child2 ) );
        EXPECT_EQ( child1.get(), child2->parent() );
        EXPECT_EQ( child1->children().size(), 1 );

        EXPECT_TRUE( root.removeChild( child1 ) );
        EXPECT_FALSE( root.removeChild( child1 ) );
        EXPECT_EQ( nullptr, child1->parent() );
        EXPECT_EQ( root.children().size(), 0 );
    }

    auto parent = child2->parent();
    EXPECT_EQ( parent, nullptr );
}

TEST( MRMesh, ObjectCloneAndSwap )
{
    Object a;
    a.setName( "a" );
    a.setXf( AffineXf3f::translation( { 1.f, 2.f, 3.f } ) );
    a.select( true );
    a.setLocked( true );

    auto clone = a.clone();
    EXPECT_EQ( clone->name(), "a" );
    EXPECT_EQ( clone->xf(), a.xf() );
    EXPECT_TRUE( clone->isSelected() );
    EXPECT_TRUE( clone->isLocked() );

    Object b;
    b.setName( "b" );
    a.swap( b );
    EXPECT_EQ( a.name(), "b" );
    EXPECT_EQ( b.name(), "a" );
    EXPECT_EQ( a.xf(), AffineXf3f() );
    EXPECT_EQ( b.xf(), clone->xf() );
    EXPECT_FALSE( a.isSelected() );
    EXPECT_TRUE( b.isSelected() );
}

} //namespace MR
