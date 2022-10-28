#include "MRMesh/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshIntersect.h"
#include "MRMesh/MRIntersectionPrecomputes.h"
#include "MRMesh/MRLine3.h"
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

    m.def( "rayMeshIntersect", ( std::optional<MR::MeshIntersectionResult>( * )( const MR::MeshPart&, const MR::Line3f&,
        float, float, const MR::IntersectionPrecomputes<float>*, bool ) )& MR::rayMeshIntersect,
        pybind11::arg( "meshPart" ), pybind11::arg( "line" ),
        pybind11::arg( "rayStart" ) = 0.0f, pybind11::arg( "rayEnd" ) = FLT_MAX,
        pybind11::arg( "prec" ) = nullptr, pybind11::arg( "closestIntersect" ) = true,
        "Finds ray and mesh intersection in float-precision.\n"
        "rayStart and rayEnd define the interval on the ray to detect an intersection.\n"
        "prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).\n"
        "Finds the closest to ray origin intersection (or any intersection for better performance if !closestIntersect)." );
   
    m.def( "rayMeshIntersect", ( std::optional<MR::MeshIntersectionResult>( * )( const MR::MeshPart&, const MR::Line3d&,
        double, double, const MR::IntersectionPrecomputes<double>*, bool ) )& MR::rayMeshIntersect,
        pybind11::arg( "meshPart" ), pybind11::arg( "line" ),
        pybind11::arg( "rayStart" ) = 0.0, pybind11::arg( "rayEnd" ) = DBL_MAX,
        pybind11::arg( "prec" ) = nullptr, pybind11::arg( "closestIntersect" ) = true,
        "Finds ray and mesh intersection in double-precision.\n"
        "rayStart and rayEnd define the interval on the ray to detect an intersection.\n"
        "prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).\n"
        "Finds the closest to ray origin intersection (or any intersection for better performance if !closestIntersect)." );
} )
