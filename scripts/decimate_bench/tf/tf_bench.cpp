// trueform side of the decimation comparison, the same calls as trueform's benchmarks/remesh/decimation-tf-impl.cpp
//   tf_bench time <mesh.stl> <ratio> <reps> [min_quality]       -> CSV: halfedges_ms,decimate_ms,faces   (minimum over reps)
//   tf_bench save <mesh.stl> <ratio> <min_quality> <out.stl>    -> decimates once and saves the result
// min_quality defaults to trueform's own default (-1 = triangle-quality check off)
#include <trueform/trueform.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

static double msSince( std::chrono::steady_clock::time_point t0 )
{
    return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
}

int main( int argc, char** argv )
{
    if ( argc < 5 )
    {
        std::fprintf( stderr, "usage: tf_bench time|save ...\n" );
        return 1;
    }
    const std::string mode = argv[1];
    auto mesh = tf::read_stl<int>( argv[2] );
    auto polys = mesh.polygons();
    const float ratio = float( std::atof( argv[3] ) );
    tf::decimate_config<float> config;

    if ( mode == "save" )
    {
        config.min_quality = float( std::atof( argv[4] ) );
        tf::half_edges<int> he( polys );
        auto [result, heOut] = tf::decimated( polys | tf::tag( he ), ratio, config );
        tf::write_stl( result.polygons(), argv[5] );
        return 0;
    }

    const int reps = std::atoi( argv[4] );
    if ( argc > 5 )
        config.min_quality = float( std::atof( argv[5] ) );
    double bestHe = 1e30, bestDec = 1e30;
    std::size_t faces = 0;
    for ( int i = 0; i < reps; ++i )
    {
        auto t0 = std::chrono::steady_clock::now();
        tf::half_edges<int> he( polys );
        bestHe = std::min( bestHe, msSince( t0 ) );
        auto tagged = polys | tf::tag( he );
        t0 = std::chrono::steady_clock::now();
        auto [result, heOut] = tf::decimated( tagged, ratio, config );
        bestDec = std::min( bestDec, msSince( t0 ) );
        faces = result.polygons().size();
    }
    std::printf( "%.2f,%.2f,%zu\n", bestHe, bestDec, faces );
    return 0;
}
