#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBoolean.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

int main( int argc, char* argv[] )
{
    // the number of Booleans can be given in the command line
    const int numBooleans = argc > 1 ? std::atoi( argv[1] ) : 2000;

    // two identical spheres of radius 1
    MR::Mesh sphereA = MR::makeSphere( { .radius = 1.0f, .numMeshVertices = 3366 } );
    MR::Mesh sphereB = sphereA;

    // subtracts sphereB rotated around Z-axis by i * 0.1 degrees from sphereA;
    // sphereB itself is not modified, so the AABB trees of both meshes are built once and reused
    auto subtractRotated = [&] ( int i )
    {
        const auto rigidB2A = MR::AffineXf3f::linear( MR::Matrix3f::rotation( MR::Vector3f::plusZ(), i * 0.1f * MR::PI_F / 180 ) );
        return MR::boolean( sphereA, sphereB, MR::BooleanOperation::DifferenceAB, { .rigidB2A = &rigidB2A } );
    };

    // untimed warm-up Boolean, it builds the AABB trees
    if ( auto res = subtractRotated( 1 ); !res.valid() )
    {
        std::cerr << res.errorString << std::endl;
        return 1;
    }

    const auto start = std::chrono::steady_clock::now();
    for ( int i = 1; i <= numBooleans; ++i )
    {
        if ( auto res = subtractRotated( i ); !res.valid() )
        {
            std::cerr << res.errorString << std::endl;
            return 1;
        }
    }
    const double seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();

    std::cout << std::fixed << numBooleans << " Booleans in " << std::setprecision( 3 ) << seconds << " s, "
              << std::setprecision( 1 ) << numBooleans / seconds << " Booleans/s" << std::endl;
    return 0;
}
