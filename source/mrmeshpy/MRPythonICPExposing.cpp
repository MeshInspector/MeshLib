#include "MRPython/MRPython.h"
#include "MRMesh/MRICP.h"
#include "MRMesh/MRMesh.h"

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, PointPair, MR::PointPair )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, PointPairs, MR::PointPairs )

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
        def_readwrite( "srcPoint", &MR::PointPair::srcPoint, "coordinates of the source point after transforming in world space" ).
        def_readwrite( "srcNorm", &MR::PointPair::srcNorm, "normal in source point after transforming in world space" ).
        def_readwrite( "tgtCloseVert", &MR::PointPair::tgtCloseVert, "for point clouds it is the closest vertex on target, for meshes it is the closest vertex of the triangle with the closest point on target" ).
        def_readwrite( "tgtPoint", &MR::PointPair::tgtPoint, "coordinates of the closest point on target after transforming in world space" ).
        def_readwrite( "tgtNorm", &MR::PointPair::tgtNorm, "normal in the target point after transforming in world space" ).
        def_readwrite( "normalsAngleCos", &MR::PointPair::normalsAngleCos, "cosine between normals in source and target points" ).
        def_readwrite( "distSq", &MR::PointPair::distSq, "squared distance between source and target points" ).
        def_readwrite( "weight", &MR::PointPair::weight, "weight of the pair (to prioritize over other pairs)" ).
        def_readwrite( "tgtOnBd", &MR::PointPair::tgtOnBd, "true if if the closest point on target is located on the boundary (only for meshes)" );

    MR_PYTHON_CUSTOM_CLASS( PointPairs ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &MR::PointPairs::vec, "vector of all point pairs both active and not" ).
        def_readwrite( "active", &MR::PointPairs::active, "whether corresponding pair from vec must be considered during minimization" );

    pybind11::class_<MR::ICPProperties>( m, "ICPProperties" ).
        def( pybind11::init<>() ).
        def_readwrite( "method", &MR::ICPProperties::method, "The method how to update transformation from point pairs" ).
        def_readwrite( "p2plAngleLimit", &MR::ICPProperties::p2plAngleLimit, "Rotation angle during one iteration of PointToPlane will be limited by this value").
        def_readwrite( "p2plScaleLimit", &MR::ICPProperties::p2plScaleLimit, "Scaling during one iteration of PointToPlane will be limited by this value").
        def_readwrite( "cosThreshold", &MR::ICPProperties::cosThreshold, "Points pair will be counted only if cosine between surface normals in points is higher" ).
        def_readwrite( "distThresholdSq", &MR::ICPProperties::distThresholdSq, "Points pair will be counted only if squared distance between points is lower than" ).
        def_readwrite( "farDistFactor", &MR::ICPProperties::farDistFactor,
            "Points pair will be counted only if distance between points is lower than root-mean-square distance times this factor" ).
        def_readwrite( "icpMode", &MR::ICPProperties::icpMode, "Finds only translation. Rotation part is identity matrix" ).
        def_readwrite( "fixedRotationAxis", &MR::ICPProperties::fixedRotationAxis, "If this vector is not zero then rotation is allowed relative to this axis only" ).
        def_readwrite( "iterLimit", &MR::ICPProperties::iterLimit, "maximum iterations" ).
        def_readwrite( "badIterStopCount", &MR::ICPProperties::badIterStopCount, "maximum iterations without improvements" ).
        def_readwrite( "exitVal", &MR::ICPProperties::exitVal, "Algorithm target root-mean-square distance. As soon as it is reached, the algorithm stops." ).
        def_readwrite( "mutualClosest", &MR::ICPProperties::mutualClosest, "A pair of points is formed only if both points in the pair are mutually closest (reciprocity test passed)." );
        
    pybind11::class_<MR::ICP>( m, "ICP", "This class allows to match two meshes with almost same geometry throw ICP point-to-point or point-to-plane algorithms" ).
        def( pybind11::init<const MR::Mesh&, const MR::Mesh&, const MR::AffineXf3f&, const MR::AffineXf3f&, const MR::VertBitSet&, const MR::VertBitSet&>(),
            pybind11::arg( "flt" ), pybind11::arg( "ref" ),
            pybind11::arg( "fltXf" ), pybind11::arg( "refXf" ),
            // pybind11::arg( "fltSamples" ) = MR::VertBitSet{}, does not work for pybind11-stubgen "Invalid expression"
            pybind11::arg_v( "fltSamples", MR::VertBitSet(), "VertBitSet()" ),
            pybind11::arg_v( "refSamples", MR::VertBitSet(), "VertBitSet()" ),
            "Constructs ICP framework with given sample points on both objects\n"
            "flt - floating object\n"
            "ref - reference object\n"
            "fltXf - transformation from floating object space to global space\n"
            "refXf - transformation from reference object space to global space\n"
            "fltSamples - samples on floating object to find projections on the reference object during the algorithm\n"
            "refSamples - samples on reference object to find projections on the floating object during the algorithm" ).
        def( pybind11::init<const MR::Mesh&, const MR::Mesh&, const MR::AffineXf3f&, const MR::AffineXf3f&, float>(), 
            pybind11::arg( "flt" ), pybind11::arg( "ref" ),
            pybind11::arg( "fltXf" ), pybind11::arg( "refXf" ),
            pybind11::arg( "samplingVoxelSize" ),
            "Constructs ICP framework with automatic points sampling on both objects\n"
            "flt - floating object\n"
            "ref - reference object\n"
            "fltXf - transformation from floating object space to global space\n"
            "refXf - transformation from reference object space to global space\n"
            "samplingVoxelSize - approximate distance between samples on each of two objects" ).
        def( "setParams", &MR::ICP::setParams, pybind11::arg( "prop" ), "tune algorithm params before run calculateTransformation()" ).
        def( "setCosineLimit", &MR::ICP::setCosineLimit, pybind11::arg( "cos" ) ).
        def( "setDistanceLimit", &MR::ICP::setDistanceLimit, pybind11::arg( "dist" ) ).
        def( "setBadIterCount", &MR::ICP::setBadIterCount, pybind11::arg( "iter" ) ).
        def( "setFarDistFactor", &MR::ICP::setFarDistFactor, pybind11::arg( "factor" ) ).
        def( "sampleFltPoints", &MR::ICP::sampleFltPoints, pybind11::arg( "samplingVoxelSize" ) ).
        def( "sampleRefPoints", &MR::ICP::sampleRefPoints, pybind11::arg( "samplingVoxelSize" ) ).
        def( "samplePoints", &MR::ICP::samplePoints, pybind11::arg( "samplingVoxelSize" ) ).
        def( "recomputeBitSet", &MR::ICP::sampleFltPoints, pybind11::arg( "floatSamplingVoxelSize" ) ).
        def( "getParams", &MR::ICP::getParams, pybind11::return_value_policy::copy ).
        def( "getLastICPInfo", &MR::ICP::getStatusInfo, "returns status info string (old function, please use .getStatusInfo())" ).
        def( "getStatusInfo", &MR::ICP::getStatusInfo, "returns status info string" ).
        def( "getNumActivePairs", &MR::ICP::getNumActivePairs, "computes the number of active point pairs" ).
        def( "getMeanSqDistToPoint", &MR::ICP::getMeanSqDistToPoint, "computes root-mean-square deviation between points" ).
        def( "getMeanSqDistToPlane", &MR::ICP::getMeanSqDistToPlane, "computes root-mean-square deviation from points to target planes" ).
        def( "getFlt2RefPairs", &MR::ICP::getFlt2RefPairs, pybind11::return_value_policy::copy, "returns current pairs formed from samples on floating object and projections on reference object" ).
        def( "getRef2FltPairs", &MR::ICP::getRef2FltPairs, pybind11::return_value_policy::copy, "returns current pairs formed from samples on reference object and projections on floating object" ).
        def( "calculateTransformation", &MR::ICP::calculateTransformation, "runs ICP algorithm given input objects, transformations, and parameters; "
            "returns adjusted transformation of the floating object to match reference object" ).
        def( "autoSelectFloatXf", &MR::ICP::autoSelectFloatXf, "automatically selects initial transformation for the floating object based on covariance matrices of both floating and reference objects; applies the transformation to the floating object and returns it" ).
        def( "updatePointPairs", &MR::ICP::updatePointPairs, "recompute point pairs after manual change of transformations or parameters" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorICPPointPair, MR::PointPair )
