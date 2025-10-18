#include "MRMesh/MRObject.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRMesh/MRObjectSave.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRGTest.h"

namespace MR
{

TEST( MRMesh, SerializeObject )
{
    Object o;
    o.setName( "123" );
    UniqueTemporaryFolder f({});
    auto mruPath = f / "123.mru";
    auto s = serializeObjectTree( o, mruPath );
    EXPECT_TRUE( s.has_value() );
    auto l = loadSceneFromAnySupportedFormat( mruPath );
    EXPECT_TRUE( l.has_value() );
    EXPECT_TRUE( l->obj.get() );
    EXPECT_TRUE( l->obj->name() == o.name() );
    EXPECT_TRUE( l->warnings.empty() );
}

} //namespace MR
