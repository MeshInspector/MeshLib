#include "mrmeshpy/MRPython.h"
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
        value( "Combined", MR::ICPMethod::Combined ).
        value( "PointToPoint", MR::ICPMethod::PointToPoint ).
        value( "PointToPlane", MR::ICPMethod::PointToPlane );
    
    pybind11::enum_<MR::ICPMode>( m, "ICPMode" ).
        value( "AnyRigidXf", MR::ICPMode::AnyRigidXf ).
        value( "OrthogonalAxis", MR::ICPMode::OrthogonalAxis ).
        value( "FixedAxis", MR::ICPMode::FixedAxis ).
        value( "TranslationOnly", MR::ICPMode::TranslationOnly );

    pybind11::class_<MR::VertPair>( m, "ICPVertPair" ).
        def( pybind11::init<>() ).
        def_readwrite( "refPoint", &MR::VertPair::refPoint ).
        def_readwrite( "norm", &MR::VertPair::norm ).
        def_readwrite( "normRef", &MR::VertPair::normRef ).
        def_readwrite( "vertId", &MR::VertPair::vertId ).
        def_readwrite( "normalsAngleCos", &MR::VertPair::normalsAngleCos ).
        def_readwrite( "vertDist2", &MR::VertPair::vertDist2 ).
        def_readwrite( "weight", &MR::VertPair::weight );

    pybind11::class_<MR::ICPProperties>( m, "ICPProperties" ).
        def( pybind11::init<>() ).
        def_readwrite( "method", &MR::ICPProperties::method ).
        def_readwrite( "p2plAngleLimit", &MR::ICPProperties::p2plAngleLimit ).
        def_readwrite( "cosTreshold", &MR::ICPProperties::cosTreshold ).
        def_readwrite( "distTresholdSq", &MR::ICPProperties::distTresholdSq ).
        def_readwrite( "distStatisticSigmaFactor ", &MR::ICPProperties::distStatisticSigmaFactor ).
        def_readwrite( "icpMode", &MR::ICPProperties::icpMode ).
        def_readwrite( "fixedRotationAxis", &MR::ICPProperties::fixedRotationAxis ).
        def_readwrite( "freezePairs", &MR::ICPProperties::freezePairs ).
        def_readwrite( "iterLimit", &MR::ICPProperties::iterLimit ).
        def_readwrite( "badIterStopCount", &MR::ICPProperties::badIterStopCount ).
        def_readwrite( "exitVal", &MR::ICPProperties::exitVal );
        
    pybind11::class_<MR::MeshICP>( m, "MeshICP" ).
        def( pybind11::init<const MR::Mesh&, const MR::Mesh&, const MR::AffineXf3f&, const MR::AffineXf3f&, const MR::VertBitSet&>() ).
        def( pybind11::init<const MR::Mesh&, const MR::Mesh&, const MR::AffineXf3f&, const MR::AffineXf3f&, float>() ).
        def( "setParams", &MR::MeshICP::setParams ).
        def( "setCosineLimit", &MR::MeshICP::setCosineLimit ).
        def( "setDistanceLimit", &MR::MeshICP::setDistanceLimit ).
        def( "setBadIterCount", &MR::MeshICP::setBadIterCount ).
        def( "setPairsWeight", &MR::MeshICP::setPairsWeight ).
        def( "setDistanceFilterSigmaFactor", &MR::MeshICP::setDistanceFilterSigmaFactor ).
        def( "recomputeBitSet", &MR::MeshICP::recomputeBitSet ).
        def( "getParams", &MR::MeshICP::getParams,pybind11::return_value_policy::copy ).
        def( "getShiftVector", &MR::MeshICP::getShiftVector ).
        def( "getLastICPInfo", &MR::MeshICP::getLastICPInfo ).
        def( "getMeanSqDistToPoint", &MR::MeshICP::getMeanSqDistToPoint ).
        def( "getMeanSqDistToPlane", &MR::MeshICP::getMeanSqDistToPlane ).
        def( "getVertPairs", &MR::MeshICP::getVertPairs, pybind11::return_value_policy::copy ).
        def( "getDistLimitsSq", &MR::MeshICP::getDistLimitsSq ).
        def( "calculateTransformation", &MR::MeshICP::calculateTransformation ).
        def( "updateVertPairs", &MR::MeshICP::updateVertPairs );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorICPVertPair, MR::VertPair )
