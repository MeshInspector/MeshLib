#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRObject.h>
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, DataModel)
{
    Object root;
    EXPECT_EQ(root.children().size(), 0);

    auto child = std::make_shared<Object>();
    EXPECT_TRUE(root.addChild(child));
    EXPECT_FALSE(root.addChild(child));
    EXPECT_EQ(&root, child->parent());
    EXPECT_EQ(root.children().size(), 1);

    child->setName( "child" );
    EXPECT_EQ( child, root.find( "child" ) );
    EXPECT_FALSE( root.find( "something" ) );
    EXPECT_EQ( child, root.find<Object>( "child" ) );
    EXPECT_FALSE( root.find<ObjectMesh>( "child" ) );

    auto grandchild = std::make_shared<ObjectMesh>();
    EXPECT_TRUE(child->addChild(grandchild));
    EXPECT_EQ(child.get(), grandchild->parent());

    EXPECT_TRUE(root.removeChild(child));
    EXPECT_FALSE(root.removeChild(child));
    EXPECT_EQ(nullptr, child->parent());
    EXPECT_EQ(root.children().size(), 0);

    child->removeAllChildren();
    EXPECT_EQ(child->children().size(), 0);
}

} //namespace MR
