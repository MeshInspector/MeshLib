#include "MRPython/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshIntersect.h"
#include "MRMesh/MRMeshThickness.h"
#include "MRMesh/MRIntersectionPrecomputes.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRPointsToMeshProjector.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include <pybind11/stl.h>

using namespace MR;

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshIntersect, [] ( pybind11::module_& m )
{
    pybind11::class_<MeshIntersectionResult>( m, "MeshIntersectionResult" ).
        def( pybind11::init<>() ).
        def( "__bool__", &MeshIntersectionResult::operator bool ).
        def_readwrite( "proj", &MeshIntersectionResult::proj, "stores intersected face and global coordinates" ).
        def_readwrite( "mtp", &MeshIntersectionResult::mtp, "stores barycentric coordinates" ).
        def_readwrite( "distanceAlongLine", &MeshIntersectionResult::distanceAlongLine,
            "stores the distance from ray origin to the intersection point in direction units" );

    pybind11::class_<IntersectionPrecomputes<float>>( m, "IntersectionPrecomputesf",
        "stores useful precomputed values for presented direction vector\n"
        "allows to avoid repeatable computations during intersection finding" ).
        def( pybind11::init<const Vector3f&>(), pybind11::arg( "dir" ) );

    pybind11::class_<IntersectionPrecomputes<double>>(m, "IntersectionPrecomputesd",
        "stores useful precomputed values for presented direction vector\n"
        "allows to avoid repeatable computations during intersection finding" ).
        def( pybind11::init<const Vector3d&>(), pybind11::arg( "dir" ) );

    m.def( "rayMeshIntersect", []( const MeshPart& meshPart, const Line3f& line,
            float rayStart, float rayEnd, const IntersectionPrecomputes<float>* prec, bool closestIntersect )
            { return rayMeshIntersect( meshPart, line, rayStart, rayEnd, prec, closestIntersect ); },
        pybind11::arg( "meshPart" ), pybind11::arg( "line" ),
        pybind11::arg( "rayStart" ) = 0.0f, pybind11::arg( "rayEnd" ) = FLT_MAX,
        pybind11::arg( "prec" ) = nullptr, pybind11::arg( "closestIntersect" ) = true,
        "Finds ray and mesh intersection in float-precision.\n"
        "rayStart and rayEnd define the interval on the ray to detect an intersection.\n"
        "prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).\n"
        "Finds the closest to ray origin intersection (or any intersection for better performance if !closestIntersect)." );
   
    m.def( "rayMeshIntersect", []( const MeshPart& meshPart, const Line3d& line,
            double rayStart, double rayEnd, const IntersectionPrecomputes<double>* prec, bool closestIntersect )
            { return rayMeshIntersect( meshPart, line, rayStart, rayEnd, prec, closestIntersect ); },
        pybind11::arg( "meshPart" ), pybind11::arg( "line" ),
        pybind11::arg( "rayStart" ) = 0.0, pybind11::arg( "rayEnd" ) = DBL_MAX,
        pybind11::arg( "prec" ) = nullptr, pybind11::arg( "closestIntersect" ) = true,
        "Finds ray and mesh intersection in double-precision.\n"
        "rayStart and rayEnd define the interval on the ray to detect an intersection.\n"
        "prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).\n"
        "Finds the closest to ray origin intersection (or any intersection for better performance if !closestIntersect)." );
    
    m.def( "computeRayThicknessAtVertices", []( const Mesh& mesh ) { return *computeRayThicknessAtVertices( mesh ); },
        pybind11::arg( "mesh" ),
        "Returns the distance from each vertex along minus normal to the nearest mesh intersection.\n"
        "Returns FLT_MAX if no intersection found)\n" );
    // deprecated
    m.def( "computeThicknessAtVertices", []( const Mesh& mesh ) { return *computeRayThicknessAtVertices( mesh ); },
        pybind11::arg( "mesh" ),
        "Returns the distance from each vertex along minus normal to the nearest mesh intersection.\n"
        "Returns FLT_MAX if no intersection found)\n" );
    
    pybind11::class_<InSphereSearchSettings>( m, "InSphereSearchSettings" ).
        def( pybind11::init<>() ).
        def_readwrite( "insideAndOutside", &InSphereSearchSettings::insideAndOutside,
            "if false then searches for the maximal inscribed sphere in mesh; "
            "if true then searches for both a) maximal inscribed sphere, and b) maximal sphere outside the mesh touching it at two points; "
            "and returns the smaller of two, and if it is b) then with minus sign" ).
        def_readwrite( "maxRadius", &InSphereSearchSettings::maxRadius, "maximum allowed radius of the sphere; for almost closed meshes the article recommends maxRadius = 0.5f * std::min( { boxSize.x, boxSize.y, boxSize.z } )" ).
        def_readwrite( "maxIters", &InSphereSearchSettings::maxIters, "maximum number of shrinking iterations for one triangle" ).
        def_readwrite( "minShrinkage", &InSphereSearchSettings::minShrinkage, "iterations stop if next radius is larger than minShrinkage times previous radius" );

    m.def( "computeInSphereThicknessAtVertices", []( const Mesh& mesh, const InSphereSearchSettings & settings ) { return *computeInSphereThicknessAtVertices( mesh, settings ); },
        pybind11::arg( "mesh" ),
        pybind11::arg( "settings" ),
        "Returns the thickness at each vertex as the diameter of the inscribed sphere." );
} )

namespace
{

std::vector<float> projectAllPoints( const Mesh& refMesh, const std::vector<Vector3f>& points, const AffineXf3f* refXf = nullptr, const AffineXf3f* xf = nullptr, float upDistLimitSq = FLT_MAX, float loDistLimitSq = 0.0f )
{
    PointsToMeshProjector projector;
    projector.updateMeshData( &refMesh );
    std::vector<MeshProjectionResult> mpRes( points.size() );
    projector.findProjections( mpRes, points, xf, refXf, upDistLimitSq, loDistLimitSq );
    std::vector<float> res( points.size(), std::sqrt( upDistLimitSq ) );

    AffineXf3f fullXf;
    if ( refXf )
        fullXf = refXf->inverse();
    if ( xf )
        fullXf = fullXf * ( *xf );

    ParallelFor( points, [&] ( size_t i )
    {
        const auto& mpResV = mpRes[i];
        auto& resV = res[i];

        resV = mpResV.distSq;
        if ( mpResV.mtp.e )
            resV = refMesh.signedDistance( fullXf( points[i] ), mpResV );
        else
            resV = std::sqrt( resV );
    } );
    return res;
}

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
            resV = refMesh.signedDistance( fullXf( mesh.points[v] ), mpResV );
        else
            resV = std::sqrt( resV );
    } );
    return res;
}

}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshProjection, [] ( pybind11::module_& m )
{
    pybind11::class_<MeshProjectionResult>( m, "MeshProjectionResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "proj", &MeshProjectionResult::proj, "the closest point on mesh, transformed by xf if it is given" ).
        def_readwrite( "mtp", &MeshProjectionResult::mtp, "its barycentric representation" ).
        def_readwrite( "distSq", &MeshProjectionResult::distSq, "squared distance from pt to proj" );

    m.def( "findProjection", []( const Vector3f & pt, const MeshPart & mp, float upDistLimitSq, const AffineXf3f * xf, float loDistLimitSq )
        { return findProjection( pt, mp, upDistLimitSq, xf, loDistLimitSq ); },
        pybind11::arg( "pt" ), pybind11::arg( "mp" ),
        pybind11::arg( "upDistLimitSq" ) = FLT_MAX, pybind11::arg( "xf" ) = nullptr,
        pybind11::arg( "loDistLimitSq" ) = 0.0f,
        "computes the closest point on mesh (or its region) to given point\n"
        "\tupDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point\n"
        "\txf mesh-to-point transformation, if not specified then identity transformation is assumed\n"
        "\tloDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one" );

    pybind11::class_<SignedDistanceToMeshResult>( m, "SignedDistanceToMeshResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "proj", &SignedDistanceToMeshResult::proj, "the closest point on mesh" ).
        def_readwrite( "mtp", &SignedDistanceToMeshResult::mtp, "its barycentric representation" ).
        def_readwrite( "dist", &SignedDistanceToMeshResult::dist, "distance from pt to proj (positive - outside, negative - inside the mesh)" );

    m.def( "findSignedDistance", &findSignedDistance,
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
        "\txf world transform for mesh\n"
        "\tupDistLimitSq upper limit on the distance in question, if the real distance is larger than the returning upDistLimit\n"
        "\tloDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one" );

    m.def( "projectAllPoints", &projectAllPoints,
        pybind11::arg( "refMesh" ), pybind11::arg( "points" ),
        pybind11::arg( "refXf" ) = nullptr, pybind11::arg( "xf" ) = nullptr,
        pybind11::arg( "upDistLimitSq" ) = FLT_MAX, pybind11::arg( "loDistLimitSq" ) = 0.0f,
        "computes signed distances from all points to refMesh\n"
        "\trefMesh all points will me projected to this mesh\n"
        "\tpoints will be projected\n"
        "\trefXf world transform for refMesh\n"
        "\txf world transform for points\n"
        "\tupDistLimitSq upper limit on the distance in question, if the real distance is larger than the returning upDistLimit\n"
        "\tloDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one" );
} )
