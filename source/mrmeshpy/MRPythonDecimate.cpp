#include "MRMesh/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshDecimate.h"

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Decimate, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::DecimateStrategy>( m, "DecimateStrategy", "Defines the order of edge collapses inside Decimate algorithm" ).
        value( "MinimizeError", MR::DecimateStrategy::MinimizeError, "the next edge to collapse will be the one that introduced minimal error to the surface" ).
        value( "ShortestEdgeFirst", MR::DecimateStrategy::ShortestEdgeFirst, "the next edge to collapse will be the shortest one" );

    pybind11::class_<MR::DecimateSettings>( m, "DecimateSettings", "Parameters structure for decimateMesh" ).
        def( pybind11::init<>() ).
        def_readwrite( "strategy", &MR::DecimateSettings::strategy ).
        def_readwrite( "maxError", &MR::DecimateSettings::maxError, 
            "for DecimateStrategy::MinimizeError:\n"
            "\tstop the decimation as soon as the estimated distance deviation from the original mesh is more than this value\n"
            "for DecimateStrategy::ShortestEdgeFirst only:\n"
            "\tstop the decimation as soon as the shortest edge in the mesh is greater than this value" ).
        def_readwrite( "maxEdgeLen", &MR::DecimateSettings::maxEdgeLen, "Maximal possible edge length created during decimation" ).
        def_readwrite( "maxDeletedFaces", &MR::DecimateSettings::maxDeletedFaces, "Limit on the number of deleted faces" ).
        def_readwrite( "maxDeletedVertices", &MR::DecimateSettings::maxDeletedVertices, "Limit on the number of deleted vertices" ).
        def_readwrite( "maxTriangleAspectRatio", &MR::DecimateSettings::maxTriangleAspectRatio, "Maximal possible aspect ratio of a triangle introduced during decimation" ).
        def_readwrite( "stabilizer", &MR::DecimateSettings::stabilizer,
            "Small stabilizer is important to achieve good results on completely planar mesh parts,\n"
            "if your mesh is not-planer everywhere, then you can set it to zero" ).
        def_readwrite("optimizeVertexPos",&MR::DecimateSettings::optimizeVertexPos,
            "if true then after each edge collapse the position of remaining vertex is optimized to\n"
            "minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position" ).
        def_readwrite( "region", &MR::DecimateSettings::region, "Region on mesh to be decimated, it is updated during the operation" ).
        def_readwrite( "touchBdVertices", &MR::DecimateSettings::touchBdVertices, "Whether to allow collapsing edges having at least one vertex on (region) boundary" ).
        def_readwrite( "packMesh", &MR::DecimateSettings::packMesh, "whether to pack mesh at the end" );

    pybind11::class_<MR::DecimateResult>( m, "DecimateResult", "Results of decimateMesh" ).
        def( pybind11::init<>() ).
        def_readwrite( "vertsDeleted", &MR::DecimateResult::vertsDeleted, "Number deleted verts. Same as the number of performed collapses" ).
        def_readwrite( "facesDeleted", &MR::DecimateResult::facesDeleted, "Number deleted faces" ).
        def_readwrite( "errorIntroduced", &MR::DecimateResult::errorIntroduced, "estimated distance deviation of decimated mesh from the original mesh" ); // only comment about default strategy

    m.def( "decimateMesh", MR::decimateMesh, pybind11::arg( "mesh" ), pybind11::arg( "settings" ) = MR::DecimateSettings{}, 
        "Collapse edges in mesh region according to the settings" );
} )
