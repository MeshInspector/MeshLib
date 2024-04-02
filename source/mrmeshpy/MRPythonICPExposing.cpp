#include "MRMesh/MRPython.h"
#include "MRMesh/MRICP.h"
#include "MRMesh/MRMesh.h"

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, PointPair, MR::PointPair )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, ICPExposing, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::ICPMethod>( m, "ICPMethod", "The method how to update transformation from point pairs" ).
        value( "Combined", MR::ICPMethod::Combined, "PointToPoint for the first 2 iterations, and PointToPlane for the remaining iterations" ).
        value( "PointToPoint", MR::ICPMethod::PointToPoint,
            "select transformation that minimizes mean squared distance between two points in each pair, "
            "it is the safest approach but can converge slowly" ).
        value( "PointToPlane", MR::ICPMethod::PointToPlane,
            "select transformation that minimizes mean squared distance between a point and a plane via the other point in each pair, "
            "converge much faster than PointToPoint in case of many good (with not all points/normals in one plane) pairs" );
    
    pybind11::enum_<MR::ICPMode>( m, "ICPMode", "The group of transformations, each with its own degrees of freedom" ).
        value( "RigidScale", MR::ICPMode::RigidScale, "rigid body transformation with uniform scaling (7 degrees of freedom)" ).
        value( "AnyRigidXf", MR::ICPMode::AnyRigidXf, "rigid body transformation (6 degrees of freedom)" ).
        value( "OrthogonalAxis", MR::ICPMode::OrthogonalAxis, "rigid body transformation with rotation except argument axis (5 degrees of freedom)" ).
        value( "FixedAxis", MR::ICPMode::FixedAxis, "rigid body transformation with rotation around given axis only (4 degrees of freedom)" ).
        value( "TranslationOnly", MR::ICPMode::TranslationOnly, "only translation (3 degrees of freedom)" );

    MR_PYTHON_CUSTOM_CLASS( PointPair ).doc() = "Stores a pair of points: one samples on the source and the closest to it on the target";
    MR_PYTHON_CUSTOM_CLASS( PointPair ).
        def( pybind11::init<>() ).
        def_readwrite( "srcVertId", &MR::PointPair::srcVertId, "id of the source point" ).
        def_readwrite( "srcNorm", &MR::PointPair::srcNorm, "normal in source point after transforming in world space" ).
        def_readwrite( "tgtPoint", &MR::PointPair::tgtPoint, "coordinates of the closest point on target after transforming in world space" ).
        def_readwrite( "tgtNorm", &MR::PointPair::tgtNorm, "normal in the target point after transforming in world space" ).
        def_readwrite( "normalsAngleCos", &MR::PointPair::normalsAngleCos, "cosine between normals in source and target points" ).
        def_readwrite( "distSq", &MR::PointPair::distSq, "squared distance between source and target points" ).
        def_readwrite( "weight", &MR::PointPair::weight, "weight of the pair (to prioritize over other pairs)" );

    pybind11::class_<MR::ICPProperties>( m, "ICPProperties" ).
        def( pybind11::init<>() ).
        def_readwrite( "method", &MR::ICPProperties::method, "The method how to update transformation from point pairs" ).
        def_readwrite( "p2plAngleLimit", &MR::ICPProperties::p2plAngleLimit, "Rotation angle during one iteration of PointToPlane will be limited by this value").
        def_readwrite( "p2plScaleLimit", &MR::ICPProperties::p2plScaleLimit, "Scaling during one iteration of PointToPlane will be limited by this value").
        def_readwrite( "cosTreshold", &MR::ICPProperties::cosTreshold, "Points pair will be counted only if cosine between surface normals in points is higher" ).
        def_readwrite( "distThresholdSq", &MR::ICPProperties::distThresholdSq, "Points pair will be counted only if squared distance between points is lower than" ).
        def_readwrite( "farDistFactor", &MR::ICPProperties::farDistFactor,
            "Points pair will be counted only if distance between points is lower than root-mean-square distance times this factor" ).
        def_readwrite( "icpMode", &MR::ICPProperties::icpMode, "Finds only translation. Rotation part is identity matrix" ).
        def_readwrite( "fixedRotationAxis", &MR::ICPProperties::fixedRotationAxis, "If this vector is not zero then rotation is allowed relative to this axis only" ).
        def_readwrite( "freezePairs", &MR::ICPProperties::freezePairs, "keep point pairs from first iteration" ).
        def_readwrite( "iterLimit", &MR::ICPProperties::iterLimit, "maximum iterations" ).
        def_readwrite( "badIterStopCount", &MR::ICPProperties::badIterStopCount, "maximum iterations without improvements" ).
        def_readwrite( "exitVal", &MR::ICPProperties::exitVal, "Algorithm target root-mean-square distance. As soon as it is reached, the algorithm stops." );
        
    pybind11::class_<MR::ICP>( m, "ICP", "This class allows to match two meshes with almost same geometry throw ICP point-to-point or point-to-plane algorithms" ).
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
        def( "setParams", &MR::ICP::setParams, pybind11::arg( "prop" ), "tune algirithm params before run calculateTransformation()" ).
        def( "setCosineLimit", &MR::ICP::setCosineLimit, pybind11::arg( "cos" ) ).
        def( "setDistanceLimit", &MR::ICP::setDistanceLimit, pybind11::arg( "dist" ) ).
        def( "setBadIterCount", &MR::ICP::setBadIterCount, pybind11::arg( "iter" ) ).
        def( "setPairsWeight", &MR::ICP::setPairsWeight, pybind11::arg( "v" ) ).
        def( "setFarDistFactor", &MR::ICP::setFarDistFactor, pybind11::arg( "factor" ) ).
        def( "recomputeBitSet", &MR::ICP::recomputeBitSet, pybind11::arg( "floatSamplingVoxelSize" ) ).
        def( "getParams", &MR::ICP::getParams, pybind11::return_value_policy::copy ).
        def( "getShiftVector", &MR::ICP::getShiftVector, "shows mean pair vector" ).
        def( "getLastICPInfo", &MR::ICP::getLastICPInfo, "returns status info string" ).
        def( "getMeanSqDistToPoint", &MR::ICP::getMeanSqDistToPoint, "computes root-mean-square deviation between points" ).
        def( "getMeanSqDistToPlane", &MR::ICP::getMeanSqDistToPlane, "computes root-mean-square deviation from points to target planes" ).
        def( "getFlt2RefPairs", &MR::ICP::getFlt2RefPairs, pybind11::return_value_policy::copy, "returns current pairs formed from samples on floating and projections on reference" ).
        def( "getDistLimitsSq", &MR::ICP::getDistLimitsSq, "finds squared minimum and maximum pairs distances" ).
        def( "calculateTransformation", &MR::ICP::calculateTransformation, "returns new xf transformation for the floating mesh, which allows to match reference mesh" ).
        def( "autoSelectFloatXf", &MR::ICP::autoSelectFloatXf, "automatically selects initial transformation for the floating object based on covariance matrices of both floating and reference objects; applies the transformation to the floating object and returns it" ).
        def( "updatePointPairs", &MR::ICP::updatePointPairs, "recompute point pairs after manual change of transformations or parameters" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorICPPointPair, MR::PointPair )
