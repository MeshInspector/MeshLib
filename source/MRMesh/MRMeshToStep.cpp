#include "MRMeshToStep.h"
#include "MRMeshToStepData.hpp"
#include "MRBox.h"
#include "MRId.h"
#include "MRMeshComponents.h"
#include "MRProgressCallback.h"
#include "MRStringConvert.h"
#include "MRTimer.h"

#include <fstream>
#include <ostream>
#include <string>

namespace MR
{

namespace MeshToStep
{

Data makeData( const Mesh& mesh, const UndirectedEdgeBitSet& faceBoundaries, const MeshToStepSettings& settings )
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
        box.include( a );
        box.include( b );
        box.include( c );
    }
    d.diag = box.valid() ? box.diagonal() : 0;
    d.tol = settings.tolerance;
    d.edgeTol = 1e-6 * d.diag;

    const auto [regMap, numRegions] = MeshComponents::getAllComponentsMap( mesh, MeshComponents::PerEdge, &faceBoundaries );
    d.sub.assign( d.nf, -1 );
    for ( FaceId f : topology.getValidFaces() )
        d.sub[f] = regMap[f];
    d.regions.resize( numRegions );
    return d;
}

} //namespace MeshToStep

Expected<MeshToStepStats> saveMeshAsStep( const Mesh& mesh, const UndirectedEdgeBitSet& faceBoundaries, std::ostream& out,
    const MeshToStepSettings& settings )
{
    MR_TIMER;
    if ( mesh.topology.numValidFaces() <= 0 )
        return unexpected( "Mesh is empty" );
    auto d = MeshToStep::makeData( mesh, faceBoundaries, settings );
    if ( !( d.diag > 0 ) )
        return unexpected( "Mesh is degenerate" );
    if ( !MeshToStep::fitRegions( d, subprogress( settings.progress, 0.0f, 0.8f ) ) )
        return unexpectedOperationCanceled();
    MeshToStep::snapBoundaries( d );
    if ( !reportProgress( settings.progress, 0.9f ) )
        return unexpectedOperationCanceled();
    std::string name;
    for ( char c : settings.name )
        name += c == '\'' || c == '\\' || (unsigned char)c < 32 ? '_' : c;
    const auto res = MeshToStep::writeBRep( d, name, out );
    if ( !out )
        return unexpected( "Stream write error" );
    reportProgress( settings.progress, 1.0f );
    return res;
}

Expected<MeshToStepStats> saveMeshAsStep( const Mesh& mesh, const UndirectedEdgeBitSet& faceBoundaries, const std::filesystem::path& file,
    const MeshToStepSettings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( "Cannot open file for writing " + utf8string( file ) );
    return saveMeshAsStep( mesh, faceBoundaries, out, settings );
}

} //namespace MR
