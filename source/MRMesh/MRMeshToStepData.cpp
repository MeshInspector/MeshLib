#include "MRMeshToStepData.hpp"
#include "MRBox.h"
#include "MRId.h"
#include "MRMeshComponents.h"
#include "MRPch/MRFmt.h"

#include <limits>

namespace MR
{

namespace MeshToStep
{

namespace
{

/// \return the maximum error of mesh coordinates caused by their representation:
/// by float type or by a small number of decimal digits if the mesh was loaded from a text file
double coordResolution( const Mesh& mesh, const Box3d& box )
{
    const double maxAbs = std::max( { std::abs( box.min.x ), std::abs( box.min.y ), std::abs( box.min.z ),
        std::abs( box.max.x ), std::abs( box.max.y ), std::abs( box.max.z ) } );
    if ( !( maxAbs > 0 ) )
        return 0;
    // the maximum number of significant decimal digits in the shortest exact notation of the coordinates
    int maxDigits = 1;
    const size_t step = std::max<size_t>( 1, mesh.points.size() / 50000 );
    for ( size_t i = 0; i < mesh.points.size() && maxDigits < 9; i += step )
        for ( int k = 0; k < 3; ++k )
        {
            std::string str = fmt::format( "{}", mesh.points[VertId( i )][k] );
            str.resize( std::min( str.size(), str.find( 'e' ) ) );
            std::erase_if( str, []( char c ) { return c < '0' || c > '9'; } );
            const auto first = str.find_first_not_of( '0' );
            const auto last = str.find_last_not_of( '0' );
            if ( first != std::string::npos )
                maxDigits = std::max( maxDigits, int( last - first ) + 1 );
        }
    const double decimalErr = 0.5 * std::pow( 10.0, std::floor( std::log10( maxAbs ) ) - maxDigits + 1 );
    return std::max( decimalErr, double( std::numeric_limits<float>::epsilon() ) * maxAbs );
}

} //anonymous namespace

Data makeData( const Mesh& mesh, const MeshToStepSettings& settings )
{
    Data d;
    d.mesh = &mesh;
    const auto& topology = mesh.topology;
    d.nf = int( topology.lastValidFace() ) + 1;
    d.pts.resize( mesh.points.size() );
    for ( int v = 0; v < (int)d.pts.size(); ++v )
        d.pts[v] = V3( mesh.points[VertId( v )] );
    d.tris.assign( d.nf, Tri{ -1, -1, -1 } );
    d.nbr.assign( d.nf, Tri{ -1, -1, -1 } );
    d.normals.resize( d.nf );
    d.centers.resize( d.nf );
    d.areas.resize( d.nf, 0.0 );
    Box3d box;
    for ( FaceId f : topology.getValidFaces() )
    {
        EdgeId e = topology.edgeWithLeft( f );
        for ( int i = 0; i < 3; ++i )
        {
            d.tris[f][i] = topology.org( e );
            d.nbr[f][i] = topology.right( e ).valid() ? int( topology.right( e ) ) : -1;
            e = topology.prev( e.sym() );
        }
        const V3& a = d.pts[d.tris[f][0]];
        const V3& b = d.pts[d.tris[f][1]];
        const V3& c = d.pts[d.tris[f][2]];
        const V3 n = cross( b - a, c - a );
        const double len = n.length();
        d.areas[f] = len / 2;
        d.normals[f] = len > 0 ? n / len : V3();
        d.centers[f] = ( a + b + c ) / 3.0;
        d.total += d.areas[f];
        box.include( a );
        box.include( b );
        box.include( c );
    }
    d.diag = box.valid() ? box.diagonal() : 0;
    d.tol = settings.tolerance > 0 ? settings.tolerance : std::max( 2.5e-6 * d.diag, 7 * coordResolution( mesh, box ) );

    // segments are connected components of faces not separated by the given edges
    const auto [segMap, numSegs] = MeshComponents::getAllComponentsMap( mesh, MeshComponents::PerEdge, settings.faceBoundaries );
    d.numSegs = numSegs;
    d.seg.assign( d.nf, -1 );
    for ( FaceId f : topology.getValidFaces() )
        d.seg[f] = segMap[f];
    d.sub.assign( d.nf, -1 );
    return d;
}

} //namespace MeshToStep

} //namespace MR
