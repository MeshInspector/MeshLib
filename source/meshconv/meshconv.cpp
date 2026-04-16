#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRObjectSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRMeshDecimate.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRConvexHull.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRLog.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRSpdlog.h"
#include "MRIOExtras/MRIOExtras.h"
#pragma warning(push)
#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning(disable: 5267) //definition of implicit copy constructor is deprecated because it has a user-provided destructor
#endif
#include <boost/program_options.hpp>
#pragma warning(pop)
#include <boost/exception/diagnostic_information.hpp>
#include <iostream>

#ifdef _WIN32
#define MC_EXIT( exitCode ) if ( waitOnExit ) system( "pause" ); return exitCode;
#else
#define MC_EXIT( exitCode ) return exitCode;
#endif

// Fix parsing std::filesystem::path with spaces (see https://github.com/boostorg/program_options/issues/69)
namespace boost
{
template <>
inline std::filesystem::path lexical_cast<std::filesystem::path, std::string>( const std::string &arg )
{
    return std::filesystem::path( arg );
}
} //namespace boost

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

        std::cout << "remeshed successfully to target edge length " << targetEdgeLen << "\n";
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


MR::Expected<MR::ObjectPtr> combineObjs( const std::vector<std::shared_ptr<MR::Object>>& objs )
{
    std::vector<std::shared_ptr<MR::Object>> objsQueue = objs;
    enum class Type
    {
        NotDefined,
        Mesh,
        Lines,
        Points
    } resType = Type::NotDefined;

    std::shared_ptr<MR::Mesh> resMeshPtr = std::make_shared<MR::Mesh>();
    std::shared_ptr<MR::Polyline3> resLinesPtr = std::make_shared<MR::Polyline3>();
    std::shared_ptr<MR::PointCloud> resPointsPtr = std::make_shared<MR::PointCloud>();

    for ( int i = 0; i < objsQueue.size(); ++i )
    {
        auto& objPtr = objsQueue[i];
        for ( auto& newPtr : objPtr->children() )
            objsQueue.push_back( newPtr );

        if ( objPtr->typeName() == MR::Object::StaticTypeName() )
            continue;

        if ( auto objMesh = std::dynamic_pointer_cast< MR::ObjectMesh >( objPtr ) )
        {
            if ( resType == Type::NotDefined )
                resType = Type::Mesh;
            else if ( resType != Type::Mesh )
                return MR::unexpected( "Error: File contains objects of different types!" );
            
            if ( !objMesh->mesh() )
                continue;

            MR::VertMap vmap;
            resMeshPtr->addMesh( *objMesh->mesh(), nullptr, &vmap );

            auto& points = resMeshPtr->points;
            const auto xf = objMesh->worldXf();
            for ( const auto v : vmap )
                if ( v.valid() )
                    points[v] = xf( points[v] );
        }
        else if ( auto objLines = std::dynamic_pointer_cast<MR::ObjectLines>( objPtr ) )
        {
            if ( resType == Type::NotDefined )
                resType = Type::Lines;
            else if ( resType != Type::Lines )
                return MR::unexpected( "Error: File contains objects of different types!" );

            if ( !objLines->polyline() )
                continue;

            MR::VertMap vmap;
            resLinesPtr->addPart( *objLines->polyline(), &vmap );

            auto& points = resLinesPtr->points;
            const auto xf = objLines->worldXf();
            for ( const auto& v : vmap )
                if ( v.valid() )
                    points[v] = xf( points[v] );
        }
        else if ( auto objPoints = std::dynamic_pointer_cast<MR::ObjectPoints>( objPtr ) )
        {
            if ( resType == Type::NotDefined )
                resType = Type::Points;
            else if ( resType != Type::Points )
                return MR::unexpected( "Error: File contains objects of different types!" );

            if ( !objPoints->pointCloud() )
                continue;

            MR::VertMap vmap;
            resPointsPtr->addPartByMask( *objPoints->pointCloud(), objPoints->pointCloud()->validPoints, { .src2tgtVerts = &vmap } );

            auto& points = resPointsPtr->points;
            const auto xf = objPoints->worldXf();
            for ( const auto v : vmap )
                if ( v.valid() )
                    points[v] = xf( points[v] );
        }
        else
        {
            return MR::unexpected( "Error: File contains unsupported objects!" );
        }
    }
    
    if ( resType == Type::Mesh )
    {
        std::shared_ptr<MR::ObjectMesh> res = std::make_shared<MR::ObjectMesh>();
        res->setMesh( resMeshPtr );
        return res;
    }
    else if ( resType == Type::Lines )
    {
        std::shared_ptr<MR::ObjectLines> res = std::make_shared<MR::ObjectLines>();
        res->setPolyline( resLinesPtr );
        return res;
    }
    else if ( resType == Type::Points )
    {
        std::shared_ptr<MR::ObjectPoints> res = std::make_shared<MR::ObjectPoints>();
        res->setPointCloud( resPointsPtr );
        return res;
    }
    return MR::unexpected( "Error: File does not contain supported object types." );
}

// can throw
static int mainInternal( int argc, char **argv )
{
    MR::loadIOExtras();

    std::filesystem::path inFilePath;
    std::filesystem::path outFilePath;
    std::string outFormat;

    namespace po = boost::program_options;
    po::options_description generalOptions( "General options" );
    generalOptions.add_options()
        ("help", "produce help message")
        ("timings", "print performance timings in the end")
        ("input-file", po::value<std::filesystem::path>( &inFilePath ), "filename of input file")
        ("output-file", po::value<std::filesystem::path>( &outFilePath ), "filename of output file")
        ("output-ext", po::value<std::string>( &outFormat ), "extension of output file \".ext\"")
        ;

    po::options_description allGeneralOptions( "All general options" );
    allGeneralOptions.add( generalOptions );

#ifdef  _WIN32
    bool waitOnExit = false;
    po::options_description hiddenOptions( "Hidden options" );
    hiddenOptions.add_options()
        ( "wait-on-exit", po::bool_switch( &waitOnExit ), "wait before closing the program" )
        ;
    allGeneralOptions.add( hiddenOptions );
#endif

    po::options_description commands( "Mesh Commands" );
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
        .options( allGeneralOptions )
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

    if ( vm.count("help") || !vm.count("input-file") )
    {
        std::cerr << 
            "meshconv is mesh file conversion utility based on MeshInspector/MeshLib\n"
            "Usage: meshconv input-file [output-file] [options]\n"
            << allCommands << "\n";
        MC_EXIT( 0 );
    }

    if ( vm.count("timings") )
    {
        // write log to console
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level( spdlog::level::trace );
        console_sink->set_pattern( "%v" );
        MR::Logger::instance().addSink( console_sink );
    }

    std::cout << "Loading " << inFilePath << "..." << std::endl;
    MR::Timer t( "Load file" );
    auto objLoadRes = MR::loadObjectFromFile( inFilePath );
    if ( !objLoadRes.has_value() )
    {
        std::cerr << "File load error: " << objLoadRes.error() << "\n";
        MC_EXIT( 1 );
    }
    std::cout << "Loaded successfully in " << t.secondsPassed().count() << "s" << std::endl;
    t.finish();

    std::vector<MR::ObjectPtr> allObjPtrs = std::move( objLoadRes->objs );
    if ( allObjPtrs.empty() )
    {
        std::cerr << "Error: No objects in the file.\n";
        MC_EXIT( 1 );
    }
    if ( allObjPtrs.size() > 1 || !allObjPtrs[0]->children().empty() )
    {
        auto combineRes = combineObjs( allObjPtrs );
        if ( combineRes.has_value() )
            allObjPtrs = { std::move( *combineRes ) };
        else
        {
            std::cerr << combineRes.error() << "\n";
            MC_EXIT( 1 );
        }
    }

    MR::ObjectPtr firstObjPtr = allObjPtrs[0];
    std::shared_ptr<MR::ObjectMesh> objMeshPtr;
    std::shared_ptr<MR::ObjectLines> objLinesPtr;
    std::shared_ptr<MR::ObjectPoints> objPointsPtr;
    if ( auto tryObjMeshPtr = std::dynamic_pointer_cast<MR::ObjectMesh>( firstObjPtr ) )
    {
        objMeshPtr = tryObjMeshPtr;
        if ( !objMeshPtr->varMesh() )
        {
            std::cerr << "Error: mesh not found!\n";
            MC_EXIT( 1 );
        }
        auto& mesh = *objMeshPtr->varMesh();

        t.restart( "MeshInfo" );
        std::cout
            << "num vertices: " << mesh.topology.numValidVerts() << "\n"
            << "num edges:    " << mesh.topology.computeNotLoneUndirectedEdges() << "\n"
            << "num faces:    " << mesh.topology.numValidFaces() << std::endl;
        t.finish();

        std::vector<std::vector<std::string>> lists;
        for ( const po::option& o : parsedCommands.options )
        {
            if ( !doCommand( o, mesh ) )
            {
                std::cerr << "Error in command : \"" << o.string_key << " " << o.value[0] << "\"\nBreak\n";
                MC_EXIT( 1 );
            }
        }
    }
    else if ( auto tryObjLinesPtr = std::dynamic_pointer_cast<MR::ObjectLines>( firstObjPtr ) )
    {
        objLinesPtr = tryObjLinesPtr;
        if ( !objLinesPtr->polyline() )
        {
            std::cerr << "Error: polyline not found!\n";
            MC_EXIT( 1 );
        }
    }
    else if ( auto tryObjPointsPtr = std::dynamic_pointer_cast<MR::ObjectPoints>( firstObjPtr ) )
    {
        objPointsPtr = tryObjPointsPtr;
        if ( !objPointsPtr->pointCloud() )
        {
            std::cerr << "Error: point cloud not found!\n";
            MC_EXIT( 1 );
        }
    }
    else
    {
        std::cerr << "Error: conversion is not supported for this file type!\n";
        MC_EXIT( 1 );
    }

    if ( outFilePath.empty() && !outFormat.empty() )
    {
        outFilePath = inFilePath;
        outFilePath.replace_extension( outFormat );
    }

    if ( !outFilePath.empty() )
    {
        std::cout << "Saving " << outFilePath << "..." << std::endl;
        t.restart( "SaveFile" );
        MR::Expected<void> saveRes;
        if ( objMeshPtr )
            saveRes = MR::MeshSave::toAnySupportedFormat( *objMeshPtr->mesh(), outFilePath);
        else if ( objLinesPtr )
            saveRes = MR::LinesSave::toAnySupportedFormat( *objLinesPtr->polyline(), outFilePath );
        else if ( objPointsPtr )
            saveRes = MR::PointsSave::toAnySupportedFormat( *objPointsPtr->pointCloud(), outFilePath );
        if ( !saveRes.has_value() )
        {
            std::cerr << "File save error: " << saveRes.error() << "\n";
            MC_EXIT( 1 );
        }
        std::cout << "Saved successfully in " << t.secondsPassed().count() << "s" << std::endl;
        t.finish();
    }

#ifdef _WIN32
    std::cout << "Peak virtual memory usage: " << MR::bytesString( MR::getProccessMemoryInfo().maxVirtual ) << std::endl;
#endif

    MC_EXIT( 0 );
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
