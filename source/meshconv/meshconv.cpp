#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMeshSave.h"
#include "MREAlgorithms/MREMeshDecimate.h"
#include "MREAlgorithms/MREBooleanOperation.h"
#include <boost/program_options.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <iostream>


// can throw
static int mainInternal( int argc, char **argv )
{
    std::filesystem::path inFilePath;
    std::filesystem::path inFilePathSecond;
    std::filesystem::path outFilePath;
    float targetEdgeLen = 0;

    namespace po = boost::program_options;
    po::options_description generalOptions( "General options" );
    generalOptions.add_options()
        ("help", "produce help message")
        ("input-file", po::value<std::filesystem::path>( &inFilePath ), "filename of input mesh")
        ("output-file,o", po::value<std::filesystem::path>( &outFilePath ), "filename of output mesh")
        ;

    po::options_description operations( "Operations" );
    operations.add_options()
        ( "remesh", po::value<float>( &targetEdgeLen )->implicit_value( targetEdgeLen ), "optional argument if positive is target edge length after remeshing" )
        ( "unite", po::value<std::filesystem::path>( &inFilePathSecond ), "unite mesh from input file and given mesh" )
        ( "subtract", po::value<std::filesystem::path>( &inFilePathSecond ), "subtract given mesh from input file mesh given mesh" )
        ( "intersect", po::value<std::filesystem::path>( &inFilePathSecond ), "intersect mesh from input file and given mesh" )
        ;

    po::options_description cmdLine( "Available options" );
    cmdLine.add( generalOptions ).add( operations );

    po::positional_options_description p;
    p.add("input-file", 1);
    p.add("output-file", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options( cmdLine ).positional(p).run(), vm);
    po::notify(vm);


    if ( vm.count("help") || !vm.count("input-file") || !vm.count("output-file") ||
        ( vm.count( "remesh" ) + vm.count( "unite" ) + vm.count( "subtract" ) + vm.count( "intersect" ) > 1) )
    {
        std::cerr << 
            "meshconv is mesh file conversion utility based on MeshInspector/MeshLib\n"
            "Usage: meshconv input-file output-file [options]\n"
            "Do not select more than one operation\n"
            << cmdLine << "\n";
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


    if ( vm.count( "remesh" ) )
    {
        if ( targetEdgeLen <= 0 )
            targetEdgeLen = mesh.averageEdgeLength();

        MRE::RemeshSettings rems;
        rems.targetEdgeLen = targetEdgeLen;
        rems.maxDeviation = targetEdgeLen / 100;
        MRE::remesh( mesh, rems );

        std::cout << "re-meshed successfully to target edge length " << targetEdgeLen << "\n";
    }
    else
    {
        loadRes = MR::MeshLoad::fromAnySupportedFormat( inFilePathSecond );
        if ( !loadRes.has_value() )
        {
            std::cerr << "Mesh load error: " << loadRes.error() << "\n";
            return 1;
        }
        auto meshB = std::move( loadRes.value() );
        std::cout << inFilePathSecond << " loaded successfully\n";

        std::vector<MR::EdgePath> cutARes_;
        std::vector<MR::EdgePath> cutBRes_;

        auto intersectRes = MRE::findBooleanIntersections( mesh, meshB, cutARes_, cutBRes_ );
        if ( !intersectRes.empty() )
        {
            std::cerr << "Boolean intersection error: " << intersectRes << "\n";
            return 1;
        }

        MRE::BooleanOperation bo{ MRE::BooleanOperation::Union };
        if ( vm.count( "unite" ) )
            bo = MRE::BooleanOperation::Union;
        if ( vm.count( "subtract" ) )
            bo = MRE::BooleanOperation::OutsideA;
        if ( vm.count( "intersect" ) )
            bo = MRE::BooleanOperation::Intersection;
        auto resMesh = doBooleanOperation( mesh, meshB, cutARes_, cutBRes_, bo );

        if ( !resMesh.has_value() )
        {
            std::cerr << resMesh.error() << "\n";
            return 1;
        }
        else
        {
            std::cout << "Success!\n";
            mesh = std::move( resMesh.value() );
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
