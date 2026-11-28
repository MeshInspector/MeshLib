#include "MRMeshToStep.h"
#include "MRMeshToStepData.hpp"
#include "MRProgressCallback.h"
#include "MRStringConvert.h"
#include "MRTimer.h"

#include <fstream>
#include <ostream>
#include <string>

namespace MR
{

Expected<MeshToStepStats> saveMeshAsStep( const Mesh& mesh, std::ostream& out, const MeshToStepSettings& settings )
{
    MR_TIMER;
    if ( mesh.topology.numValidFaces() <= 0 )
        return unexpected( "Mesh is empty" );
    auto d = MeshToStep::makeData( mesh, settings );
    if ( !( d.tol > 0 ) )
        return unexpected( "Mesh is degenerate" );
    if ( !MeshToStep::findAnalyticRegions( d, subprogress( settings.progress, 0.0f, 0.8f ) ) )
        return unexpectedOperationCanceled();
    MeshToStep::cleanRegions( d );
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

Expected<MeshToStepStats> saveMeshAsStep( const Mesh& mesh, const std::filesystem::path& file, const MeshToStepSettings& settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( "Cannot open file for writing " + utf8string( file ) );
    return saveMeshAsStep( mesh, out, settings );
}

} //namespace MR
