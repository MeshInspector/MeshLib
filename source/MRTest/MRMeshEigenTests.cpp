#include <MRMesh/MRMeshEigen.h>
#include <MRMesh/MRMesh.h>
#include <gtest/gtest.h>

#include <MRPch/MREigenCore.h>

namespace MR
{

TEST(MRMesh, Eigen)
{
    Eigen::MatrixXd V( 3, 3 );
    V( 0, 0 ) = 0; V( 0, 1 ) = 0; V( 0, 2 ) = 0;
    V( 1, 0 ) = 1; V( 1, 1 ) = 0; V( 1, 2 ) = 0;
    V( 2, 0 ) = 0; V( 2, 1 ) = 1; V( 2, 2 ) = 0;

    Eigen::MatrixXi F( 1, 3 );
    F( 0, 0 ) = 0; F( 0, 1 ) = 1; F( 0, 2 ) = 2;

    auto mesh = meshFromEigen( V, F );

    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    meshToEigen( mesh, V1, F1 );

    EXPECT_TRUE( V == V1 );
    EXPECT_TRUE( F == F1 );
}

} //namespace MR
