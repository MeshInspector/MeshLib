#include <MRMesh/MRRegularGridMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRVector3.h>
#include <gtest/gtest.h>

namespace MR
{

TEST(MRMesh, makeRegularGridMesh)
{
     auto m = makeRegularGridMesh( 2, 2,
         []( size_t, size_t ) { return true; },
         []( size_t x, size_t y ) { return Vector3f( (float)x, (float)y, 0 ); } ).value();
     ASSERT_TRUE( m.topology.checkValidity() );

     m = makeRegularGridMesh( 2, 3,
         []( size_t, size_t ) { return true; },
         []( size_t x, size_t y ) { return Vector3f( (float)x, (float)y, 0 ); } ).value();
     ASSERT_TRUE( m.topology.checkValidity() );

     m = makeRegularGridMesh( 5, 3,
         []( size_t, size_t ) { return true; },
         []( size_t x, size_t y ) { return Vector3f( (float)x, (float)y, 0 ); } ).value();
     ASSERT_TRUE( m.topology.checkValidity() );
}

} //namespace MR
