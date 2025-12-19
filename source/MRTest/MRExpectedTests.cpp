#include <MRMesh/MRGTest.h>
#include <MRMesh/MRExpected.h>
#include <MRMesh/MRVector3.h>
#include <MRPch/MRHashMap.h>

namespace MR
{

namespace
{

struct Value
{
    Vector3f diffuseColor = Vector3f::diagonal( -1.0f );
    std::string diffuseTextureFile;
};

using MyHashMap = HashMap<std::string, Value>;

void f( MyHashMap * p )
{
    EXPECT_NE( p, nullptr );
    EXPECT_TRUE( p->empty() );
    EXPECT_EQ( p->find( std::string{} ), p->end() );
}

} //anonymous namespace

TEST( MRMesh, ExpectedHashMap )
{
    Expected<MyHashMap> mhm;
    EXPECT_TRUE( mhm.has_value() );
    EXPECT_TRUE( mhm->empty() );
    EXPECT_EQ( mhm->find( std::string{} ), mhm->end() );
    f( mhm.has_value() ? &*mhm : nullptr );
}

} //namespace MR
