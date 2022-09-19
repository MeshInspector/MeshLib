#include "MRMesh/MRPython.h"
#include "MRMesh/MRICP.h"


namespace MR
{
// this is needed to expose std::vector<MR::VertPair>
bool operator == ( const VertPair& a, const VertPair& b )
{
    return
        a.norm == b.norm &&
        a.normalsAngleCos == b.normalsAngleCos &&
        a.normRef == b.normRef &&
        a.refPoint == b.refPoint &&
        a.vertDist2 == b.vertDist2 &&
        a.vertId == b.vertId &&
        a.weight == b.weight;
}
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, ICPExposing, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::ICPMethod>( m, "ICPMethod" ).
        value( "Combined", MR::ICPMethod::Combined, "PointToPoint for the first 2 iterations, PointToPlane then" ).
        value( "PointToPoint", MR::ICPMethod::PointToPoint, "use it in the cases with big differences, takes more iterations" ).
        value( "PointToPlane", MR::ICPMethod::PointToPlane, "finds solution faster in fewer iterations" );
    
    pybind11::enum_<MR::ICPMode>( m, "ICPMode", "You could fix any axis(axes) of rotation by using this modes" ).
        value( "AnyRigidXf", MR::ICPMode::AnyRigidXf, "all 6 degrees of freedom (dof)" ).
        value( "OrthogonalAxis", MR::ICPMode::OrthogonalAxis, "5 dof, except argument axis" ).
        value( "FixedAxis", MR::ICPMode::FixedAxis, "4 dof, translation and one argument axis" ).
        value( "TranslationOnly", MR::ICPMode::TranslationOnly, "3 dof, no rotation" );

    pybind11::class_<MR::VertPair>( m, "VertPair" ).
        def( pybind11::init<>() ).
        def_readwrite( "refPoint", &MR::VertPair::refPoint, "coordinates of the closest point on reference mesh (after applying refXf)" ).
        def_readwrite( "norm", &MR::VertPair::norm, "surface normal in a vertex on the floating mesh (after applying Xf)" ).
        def_readwrite( "normRef", &MR::VertPair::normRef, "surface normal in a vertex on the reference mesh (after applying Xf)" ).
        def_readwrite( "vertId", &MR::VertPair::vertId, "ID of the floating mesh vertex (usually applying Xf required)" ).
        def_readwrite( "normalsAngleCos", &MR::VertPair::normalsAngleCos,
            "This is cosine between normals in first(floating mesh) and second(reference mesh) points\n"
            "It evaluates how good is this pair" ).
        def_readwrite( "vertDist2", &MR::VertPair::vertDist2, "Storing squared distance between vertices" ).
        def_readwrite( "weight", &MR::VertPair::weight, "weight of the pair with respect to the sum of adjoining triangles square" );

    pybind11::class_<MR::ICPProperties>( m, "ICPProperties" ).
        def( pybind11::init<>() ).
        def_readwrite( "method", &MR::ICPProperties::method ).
        def_readwrite( "p2plAngleLimit", &MR::ICPProperties::p2plAngleLimit,
            "rotation part will be limited by this value. If the whole rotation exceed this value, it will be normalized to that.\n"
            "Note: PointToPlane only!").
        def_readwrite( "cosTreshold", &MR::ICPProperties::cosTreshold, "Points pair will be counted only if cosine between surface normals in points is higher" ).
        def_readwrite( "distTresholdSq", &MR::ICPProperties::distTresholdSq, "Points pair will be counted only if squared distance between points is lower than" ).
        def_readwrite( "distStatisticSigmaFactor ", &MR::ICPProperties::distStatisticSigmaFactor,
            "Sigma multiplier for statistic throw of paints pair based on the distance\n"
            "Default: all pairs in the interval the (distance = mean +- 3*sigma) are passed" ).
        def_readwrite( "icpMode", &MR::ICPProperties::icpMode, "Finds only translation. Rotation part is identity matrix" ).
        def_readwrite( "fixedRotationAxis", &MR::ICPProperties::fixedRotationAxis, "If this vector is not zero then rotation is allowed relative to this axis only" ).
        def_readwrite( "freezePairs", &MR::ICPProperties::freezePairs, "keep point pairs from first iteration" ).
        def_readwrite( "iterLimit", &MR::ICPProperties::iterLimit, "maximum iterations" ).
        def_readwrite( "badIterStopCount", &MR::ICPProperties::badIterStopCount, "maximum iterations without improvements" ).
        def_readwrite( "exitVal", &MR::ICPProperties::exitVal, "Algorithm target root-mean-square distance. As soon as it is reached, the algorithm stops." );
        
    pybind11::class_<MR::MeshICP>( m, "MeshICP", "This class allows to match two meshes with almost same geometry throw ICP point-to-point or point-to-plane algorithms" ).
        def( pybind11::init<const MR::Mesh&, const MR::Mesh&, const MR::AffineXf3f&, const MR::AffineXf3f&, const MR::VertBitSet&>(),
            pybind11::arg("floatingMesh"), pybind11::arg( "referenceMesh" ), pybind11::arg( "fltMeshXf" ), pybind11::arg( "refMeshXf" ), pybind11::arg( "floatingMeshBitSet" ),
            "xf parameters should represent current transformations of meshes\n"
            "fltMeshXf - transform from the local floatingMesh basis to the global\n"
            "refMeshXf - transform from the local referenceMesh basis to the global\n"
            "floatingMeshBitSet - allows to take exact set of vertices from the mesh" ).
        def( pybind11::init<const MR::Mesh&, const MR::Mesh&, const MR::AffineXf3f&, const MR::AffineXf3f&, float>(), 
            pybind11::arg( "floatingMesh" ), pybind11::arg( "referenceMesh" ), pybind11::arg( "fltMeshXf" ), pybind11::arg( "refMeshXf" ), pybind11::arg( "floatSamplingVoxelSize" ),
            "xf parameters should represent current transformations of meshes\n"
            "fltMeshXf - transform from the local floatingMesh basis to the global\n"
            "refMeshXf - transform from the local referenceMesh basis to the global\n"
            "floatSamplingVoxelSize = positive value here defines voxel size, and only one vertex per voxel will be selected" ).
        def( "setParams", &MR::MeshICP::setParams, pybind11::arg( "prop" ), "tune algirithm params before run calculateTransformation()" ).
        def( "setCosineLimit", &MR::MeshICP::setCosineLimit, pybind11::arg( "cos" ) ).
        def( "setDistanceLimit", &MR::MeshICP::setDistanceLimit, pybind11::arg( "dist" ) ).
        def( "setBadIterCount", &MR::MeshICP::setBadIterCount, pybind11::arg( "iter" ) ).
        def( "setPairsWeight", &MR::MeshICP::setPairsWeight, pybind11::arg( "v" ) ).
        def( "setDistanceFilterSigmaFactor", &MR::MeshICP::setDistanceFilterSigmaFactor, pybind11::arg( "factor" ) ).
        def( "recomputeBitSet", &MR::MeshICP::recomputeBitSet, pybind11::arg( "floatSamplingVoxelSize" ) ).
        def( "getParams", &MR::MeshICP::getParams, pybind11::return_value_policy::copy ).
        def( "getShiftVector", &MR::MeshICP::getShiftVector, "shows mean pair vector" ).
        def( "getLastICPInfo", &MR::MeshICP::getLastICPInfo, "returns status info string" ).
        def( "getMeanSqDistToPoint", &MR::MeshICP::getMeanSqDistToPoint, "computes root-mean-square deviation between points" ).
        def( "getMeanSqDistToPlane", &MR::MeshICP::getMeanSqDistToPlane, "computes root-mean-square deviation from points to target planes" ).
        def( "getVertPairs", &MR::MeshICP::getVertPairs, pybind11::return_value_policy::copy, "used to visualize generated points pairs" ).
        def( "getDistLimitsSq", &MR::MeshICP::getDistLimitsSq, "finds squared minimum and maximum pairs distances" ).
        def( "calculateTransformation", &MR::MeshICP::calculateTransformation, "returns new xf transformation for the floating mesh, which allows to match reference mesh" ).
        def( "updateVertPairs", &MR::MeshICP::updateVertPairs, "recompute point pairs after manual change of transformations or parameters" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorICPVertPair, MR::VertPair )
