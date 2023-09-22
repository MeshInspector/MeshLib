#include "MRMesh/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshIntersect.h"
#include "MRMesh/MRIntersectionPrecomputes.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRPointsToMeshProjector.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include <pybind11/stl.h>

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshIntersect, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshIntersectionResult>( m, "MeshIntersectionResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "proj", &MR::MeshIntersectionResult::proj, "stores intersected face and global coordinates" ).
        def_readwrite( "mtp", &MR::MeshIntersectionResult::mtp, "stores barycentric coordinates" ).
        def_readwrite( "distanceAlongLine", &MR::MeshIntersectionResult::distanceAlongLine,
            "stores the distance from ray origin to the intersection point in direction units" );

    pybind11::class_<MR::IntersectionPrecomputes<float>>( m, "IntersectionPrecomputesf",
        "stores useful precomputed values for presented direction vector\n"
        "allows to avoid repeatable computations during intersection finding" ).
        def( pybind11::init<const MR::Vector3f&>(), pybind11::arg( "dir" ) );

    pybind11::class_<MR::IntersectionPrecomputes<double>>(m, "IntersectionPrecomputesd",
        "stores useful precomputed values for presented direction vector\n"
        "allows to avoid repeatable computations during intersection finding" ).
        def( pybind11::init<const MR::Vector3d&>(), pybind11::arg( "dir" ) );

    m.def( "rayMeshIntersect", []( const MR::MeshPart& meshPart, const MR::Line3f& line,
            float rayStart, float rayEnd, const MR::IntersectionPrecomputes<float>* prec, bool closestIntersect )
            { return MR::rayMeshIntersect( meshPart, line, rayStart, rayEnd, prec, closestIntersect ); },
        pybind11::arg( "meshPart" ), pybind11::arg( "line" ),
        pybind11::arg( "rayStart" ) = 0.0f, pybind11::arg( "rayEnd" ) = FLT_MAX,
        pybind11::arg( "prec" ) = nullptr, pybind11::arg( "closestIntersect" ) = true,
        "Finds ray and mesh intersection in float-precision.\n"
        "rayStart and rayEnd define the interval on the ray to detect an intersection.\n"
        "prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).\n"
        "Finds the closest to ray origin intersection (or any intersection for better performance if !closestIntersect)." );
   
    m.def( "rayMeshIntersect", []( const MR::MeshPart& meshPart, const MR::Line3d& line,
            double rayStart, double rayEnd, const MR::IntersectionPrecomputes<double>* prec, bool closestIntersect )
            { return MR::rayMeshIntersect( meshPart, line, rayStart, rayEnd, prec, closestIntersect ); },
        pybind11::arg( "meshPart" ), pybind11::arg( "line" ),
        pybind11::arg( "rayStart" ) = 0.0, pybind11::arg( "rayEnd" ) = DBL_MAX,
        pybind11::arg( "prec" ) = nullptr, pybind11::arg( "closestIntersect" ) = true,
        "Finds ray and mesh intersection in double-precision.\n"
        "rayStart and rayEnd define the interval on the ray to detect an intersection.\n"
        "prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).\n"
        "Finds the closest to ray origin intersection (or any intersection for better performance if !closestIntersect)." );
} )

namespace
{
using namespace MR;

VertScalars projectAllMeshVertices( const Mesh& refMesh, const Mesh& mesh, const AffineXf3f* refXf = nullptr, const AffineXf3f* xf = nullptr, float upDistLimitSq = FLT_MAX, float loDistLimitSq = 0.0f )
{
    PointsToMeshProjector projector;
    projector.updateMeshData( &refMesh );
    std::vector<MeshProjectionResult> mpRes( mesh.points.vec_.size() );
    projector.findProjections( mpRes, mesh.points.vec_, xf, refXf, upDistLimitSq, loDistLimitSq );
    VertScalars res( mesh.topology.lastValidVert() + 1, std::sqrt( upDistLimitSq ) );

    AffineXf3f fullXf;
    if ( refXf )
        fullXf = refXf->inverse();
    if ( xf )
        fullXf = fullXf * ( *xf );

    BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
    {
        const auto& mpResV = mpRes[v.get()];
        auto& resV = res[v];
        
        resV = mpResV.distSq;
        if ( mpResV.mtp.e )
            resV = refMesh.signedDistance( fullXf( mesh.points[v] ), mpResV.mtp );
        else
            resV = std::sqrt( resV );
    } );
    return res;
}

}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshProjection, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshProjectionResult>( m, "MeshProjectionResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "proj", &MR::MeshProjectionResult::proj, "the closest point on mesh, transformed by xf if it is given" ).
        def_readwrite( "mtp", &MR::MeshProjectionResult::mtp, "its barycentric representation" ).
        def_readwrite( "distSq", &MR::MeshProjectionResult::distSq, "squared distance from pt to proj" );

    m.def( "findProjection", &MR::findProjection,
        pybind11::arg( "pt" ), pybind11::arg( "mp" ),
        pybind11::arg( "upDistLimitSq" ) = FLT_MAX, pybind11::arg( "xf" ) = nullptr,
        pybind11::arg( "loDistLimitSq" ) = 0.0f, pybind11::arg( "skipFace" ) = FaceId{},
        "computes the closest point on mesh (or its region) to given point\n"
        "\tupDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point\n"
        "\txf mesh-to-point transformation, if not specified then identity transformation is assumed\n"
        "\tloDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one\n"
        "\tskipFace this triangle will be skipped and never returned as a projection" );

    pybind11::class_<MR::SignedDistanceToMeshResult>( m, "SignedDistanceToMeshResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "proj", &MR::SignedDistanceToMeshResult::proj, "the closest point on mesh" ).
        def_readwrite( "mtp", &MR::SignedDistanceToMeshResult::mtp, "its barycentric representation" ).
        def_readwrite( "dist", &MR::SignedDistanceToMeshResult::dist, "distance from pt to proj (positive - outside, negative - inside the mesh)" );

    m.def( "findSignedDistance", &MR::findSignedDistance,
        pybind11::arg( "pt" ), pybind11::arg( "mp" ),
        pybind11::arg( "upDistLimitSq" ) = FLT_MAX,
        pybind11::arg( "loDistLimitSq" ) = 0.0f,
        "computes the closest point on mesh (or its region) to given point and finds the distance with sign to it( positive - outside, negative - inside the mesh )\n"
        "\tupDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exits returning nullopt\n"
        "\tloDistLimitSq low limit on the distance in question, if the real distance smaller then the function exits returning nullopt" );

    m.def( "projectAllMeshVertices", &projectAllMeshVertices,
        pybind11::arg( "refMesh" ), pybind11::arg( "mesh" ),
        pybind11::arg( "refXf" ) = nullptr, pybind11::arg( "xf" ) = nullptr,
        pybind11::arg( "upDistLimitSq" ) = FLT_MAX, pybind11::arg( "loDistLimitSq" ) = 0.0f,
        "computes signed distances from all mesh points to refMesh\n"
        "\trefMesh all points will me projected to this mesh\n"
        "\tmesh this mesh points will be projected\n"
        "\trefXf world transform for refMesh\n"
        "\tupDistLimitSq upper limit on the distance in question, if the real distance is larger than the returning upDistLimit\n"
        "\tloDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one" );
} )
