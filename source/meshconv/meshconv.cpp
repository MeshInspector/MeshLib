#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRMeshDecimate.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRConvexHull.h"
#include <boost/program_options.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <iostream>

bool doCommand( const boost::program_options::option& option, MR::Mesh& mesh )
{
    namespace po = boost::program_options;
    if ( option.string_key == "convex-hull" )
    {
        mesh = MR::makeConvexHull( mesh );
        std::cout << "convex hull computed successfully" << std::endl;
    }
    else if ( option.string_key == "remesh" )
    {
        float targetEdgeLen{ 0.f };
        if ( !option.value.empty() )
            targetEdgeLen = std::stof( option.value[0] );
        if ( targetEdgeLen <= 0 )
            targetEdgeLen = mesh.averageEdgeLength();

        MR::RemeshSettings rems;
        rems.targetEdgeLen = targetEdgeLen;
        MR::remesh( mesh, rems );

        std::cout << "re-meshed successfully to target edge length " << targetEdgeLen << "\n";
    }
    else if ( option.string_key == "unite" || option.string_key == "subtract" || option.string_key == "intersect" )
    {
        std::filesystem::path meshPath = option.value[0];

        auto loadRes = MR::MeshLoad::fromAnySupportedFormat( meshPath );
        if ( !loadRes.has_value() )
        {
            std::cerr << "Mesh load error: " << loadRes.error() << "\n";
            return false;
        }
        auto meshB = std::move( loadRes.value() );
        std::cout << meshPath << " loaded successfully\n";

        MR::BooleanOperation bo{ MR::BooleanOperation::Union };
        if ( option.string_key == "subtract" )
            bo = MR::BooleanOperation::OutsideA;
        if ( option.string_key == "intersect" )
            bo = MR::BooleanOperation::Intersection;
        auto booleanRes = MR::boolean( mesh, meshB, bo );

        if ( !booleanRes )
        {
            std::cerr << booleanRes.errorString << "\n";
            return false;
        }
        else
        {
            std::cout << option.string_key << " success!\n";
            mesh = std::move( booleanRes.mesh );
        }
    }
    return true;
}

// can throw
static int mainInternal( int argc, char **argv )
{
    std::filesystem::path inFilePath;
    std::filesystem::path outFilePath;

    namespace po = boost::program_options;
    po::options_description generalOptions( "General options" );
    generalOptions.add_options()
        ("help", "produce help message")
        ("input-file", po::value<std::filesystem::path>( &inFilePath ), "filename of input mesh")
        ("output-file", po::value<std::filesystem::path>( &outFilePath ), "filename of output mesh")
        ;

    po::options_description commands( "Commands" );
    commands.add_options()
        ( "remesh", po::value<float>()->implicit_value( 0 ), "optional argument if positive is target edge length after remeshing" )
        ( "unite", po::value<std::filesystem::path>(), "unite mesh from input file and given mesh" )
        ( "subtract", po::value<std::filesystem::path>(), "subtract given mesh from input file mesh given mesh" )
        ( "intersect", po::value<std::filesystem::path>(), "intersect mesh from input file and given mesh" )
        ( "convex-hull", "construct convex hull of input mesh" )
        ;

    po::options_description allCommands( "Available options" );
    allCommands.add( generalOptions ).add( commands );

    po::positional_options_description p;
    p.add("input-file", 1);
    p.add("output-file", 1);

    po::parsed_options parsedGeneral = po::command_line_parser( argc, argv )
        .options( generalOptions )
        .positional( p )
        .allow_unregistered()
        .run();

    std::vector<std::string> unregisteredOptions;
    for ( const auto& o : parsedGeneral.options )
    {
        if ( o.unregistered )
            unregisteredOptions.insert( unregisteredOptions.end(), o.original_tokens.begin(), o.original_tokens.end() );
    }
    
    po::variables_map vm;
    po::store(parsedGeneral, vm);
    po::notify(vm);

    po::parsed_options parsedCommands = po::command_line_parser( unregisteredOptions )
        .options( commands )
        .allow_unregistered()
        .run();

    if ( vm.count("help") || !vm.count("input-file") || !vm.count("output-file") )
    {
        std::cerr << 
            "meshconv is mesh file conversion utility based on MeshInspector/MeshLib\n"
            "Usage: meshconv input-file output-file [options]\n"
            << allCommands << "\n";
        return 1;
    }

    auto loadRes = MR::MeshLoad::fromAnySupportedFormat( inFilePath );
    if ( !loadRes.has_value() )
    {
        std::cerr << "Mesh load error: " << loadRes.error() << "\n";
        return 1;
    }
    auto mesh = std::move( loadRes.value() );
    std::cout << inFilePath << " loaded successfully\n";

    std::vector<std::vector<std::string>> lists;
    for ( const po::option& o : parsedCommands.options )
    {
        if ( !doCommand( o, mesh ) )
        {
            std::cerr << "Error in command : \""<< o.string_key << " " << o.value[0] << "\"\nBreak\n";
            return 1;
        }
    }


    auto saveRes = MR::MeshSave::toAnySupportedFormat( mesh, outFilePath );
    if ( !saveRes.has_value() )
    {
        std::cerr << "Mesh save error:" << saveRes.error() << "\n";
        return 1;
    }
    std::cout << outFilePath << " saved successfully\n";

    return 0;
}

int main( int argc, char **argv )
{
    try
    {
        return mainInternal( argc, argv );
    }
    catch ( ... )
    {
        std::cerr << "Exception: " << boost::current_exception_diagnostic_information();
        return 2;
    }
}
