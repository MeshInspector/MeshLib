#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBoolean.h"

#include <link.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string_view>

using namespace MR;

extern "C" bool mi_is_in_heap_region( const void* p ) __attribute__(( weak ));

namespace
{

using Clock = std::chrono::steady_clock;

double msSince( Clock::time_point t0 )
{
    return std::chrono::duration<double, std::milli>( Clock::now() - t0 ).count();
}

bool isMimallocPtr( const void* p )
{
    return mi_is_in_heap_region != nullptr && mi_is_in_heap_region( p );
}

} // namespace

int main()
{
    int numSharedMimallocs = 0;
    dl_iterate_phdr( []( dl_phdr_info* info, size_t, void* data )
    {
        if ( std::string_view( info->dlpi_name ).find( "libmimalloc" ) != std::string_view::npos )
            ++*static_cast<int*>( data );
        return 0;
    }, &numSharedMimallocs );
    void* probe = std::malloc( 64 );
    std::fprintf( stderr, "shared libmimalloc objects: %d, malloc from mimalloc: %d\n", numSharedMimallocs, int( isMimallocPtr( probe ) ) );
    std::free( probe );

    int failures = 0;
    // the Boolean Benchmark tool's workload
    {
        SphereParams sp;
        sp.radius = 1.0f;
        sp.numMeshVertices = 3366;
        const Mesh a = makeSphere( sp );
        const Mesh b = makeSphere( sp );
        (void)a.getAABBTree();
        (void)b.getAABBTree();
        const auto t0 = Clock::now();
        for ( int i = 0; i < 400; ++i )
        {
            const auto xf = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), float( ( i + 1 ) * 0.1 * PI / 180.0 ) ) );
            BooleanParameters params;
            params.rigidB2A = &xf;
            if ( !boolean( a, b, BooleanOperation::DifferenceAB, params ).valid() )
                ++failures;
        }
        std::printf( "spheres3366_diff_x400 %.2f\n", msSince( t0 ) );
    }
    // a large mesh minus / plus a smaller one
    {
        SphereParams sp;
        sp.radius = 1.0f;
        sp.numMeshVertices = 1000000;
        const Mesh big = makeSphere( sp );
        sp.radius = 0.7f;
        sp.numMeshVertices = 100000;
        Mesh smallMesh = makeSphere( sp );
        smallMesh.transform( AffineXf3f::translation( Vector3f( 0.6f, 0.0f, 0.0f ) ) );
        (void)big.getAABBTree();
        (void)smallMesh.getAABBTree();
        for ( auto [name, op] : { std::pair{ "big_minus_small", BooleanOperation::DifferenceAB }, std::pair{ "big_union_small", BooleanOperation::Union } } )
        {
            const auto t0 = Clock::now();
            if ( !boolean( big, smallMesh, op ).valid() )
                ++failures;
            std::printf( "%s %.2f\n", name, msSince( t0 ) );
        }
    }
    std::fprintf( stderr, "failures: %d\n", failures );
    return failures == 0 ? 0 : 1;
}
