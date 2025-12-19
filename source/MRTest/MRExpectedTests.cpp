#include <MRMesh/MRGTest.h>
#include <MRMesh/MRExpected.h>
#include <MRPch/MRHashMap.h>

namespace MR
{

TEST( MRMesh, ExpectedHashMap )
{
    using MyHashMap = HashMap<std::string, std::string>;
    Expected<MyHashMap> mhm;
    EXPECT_TRUE( mhm.has_value() );
    EXPECT_TRUE( mhm->empty() );
    EXPECT_EQ( mhm->find( std::string{} ), mhm->end() );
}

} //namespace MR
