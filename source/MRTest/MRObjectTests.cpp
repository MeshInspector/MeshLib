#include "MRMesh/MRObject.h"
#include "MRMesh/MRGTest.h"

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

} //namespace MR
