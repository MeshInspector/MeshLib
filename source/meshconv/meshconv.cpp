#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMeshSave.h"
#include <boost/program_options.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <iostream>

// can throw
static int mainInternal( int argc, char **argv )
{
    std::filesystem::path inFilePath;
    std::filesystem::path outFilePath;

    namespace po = boost::program_options;
    po::options_description desc("Available options");
    desc.add_options()
        ("help", "produce help message")
        ("input-file", po::value<std::filesystem::path>( &inFilePath ), "filename of input mesh")
        ("output-file", po::value<std::filesystem::path>( &outFilePath ), "filename of output mesh")
        ;

    po::positional_options_description p;
    p.add("input-file", 1);
    p.add("output-file", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help") || !vm.count("input-file") || !vm.count("output-file"))
    {
        std::cerr << "meshconv input-file output-file [options]" << std::endl;
        std::cerr << desc << "\n";
        return 1;
    }

    auto loadRes = MR::MeshLoad::fromAnySupportedFormat( inFilePath );
    if ( !loadRes.has_value() )
    {
        std::cerr << "Mesh load error: " << loadRes.error() << "\n";
        return 1;
    }

    auto saveRes = MR::MeshSave::toAnySupportedFormat( loadRes.value(), outFilePath );
    if ( !saveRes.has_value() )
    {
        std::cerr << "Mesh save error:" << saveRes.error() << "\n";
        return 1;
    }
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
        std::cerr << "Exceptino: " << boost::current_exception_diagnostic_information();
        return 2;
    }
}
