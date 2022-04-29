#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMeshSave.h"
#include "MREAlgorithms/MREMeshDecimate.h"
#include <boost/program_options.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <iostream>

// can throw
static int mainInternal( int argc, char **argv )
{
    std::filesystem::path inFilePath;
    std::filesystem::path outFilePath;
    float targetEdgeLen = 0;

    namespace po = boost::program_options;
    po::options_description desc( "Available options" );
    desc.add_options()
        ("help", "produce help message")
        ("input-file", po::value<std::filesystem::path>( &inFilePath ), "filename of input mesh")
        ("output-file", po::value<std::filesystem::path>( &outFilePath ), "filename of output mesh")
        ("remesh", po::value<float>( &targetEdgeLen )->implicit_value( targetEdgeLen ), "optional argument if positive is target edge length after remeshing")
        ;

    po::positional_options_description p;
    p.add("input-file", 1);
    p.add("output-file", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if ( vm.count("help") || !vm.count("input-file") || !vm.count("output-file") )
    {
        std::cerr << 
            "meshconv is mesh file conversion utility based on MeshInspector/MeshLib\n"
            "Usage: meshconv input-file output-file [options]\n"
            << desc << "\n";
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

    if ( vm.count("remesh") )
    {
        if ( targetEdgeLen <= 0 )
            targetEdgeLen = mesh.averageEdgeLength();

        MRE::RemeshSettings rems;
        rems.targetEdgeLen = targetEdgeLen;
        rems.maxDeviation = targetEdgeLen / 100;
        MRE::remesh( mesh, rems );

        std::cout << "re-meshed successfully to target edge length " << targetEdgeLen << "\n";
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
