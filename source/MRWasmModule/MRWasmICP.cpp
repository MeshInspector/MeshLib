#include "MRWasmBindings.h"

#include "MRMesh/MRICP.h"
#include "MRMesh/MRICPEnums.h"
#include "MRMesh/MRMeshOrPoints.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_icp )
{
    emscripten::enum_<ICPMethod>( "ICPMethod" )
        .value( "Combined", ICPMethod::Combined )
        .value( "PointToPoint", ICPMethod::PointToPoint )
        .value( "PointToPlane", ICPMethod::PointToPlane );

    emscripten::enum_<ICPMode>( "ICPMode" )
        .value( "RigidScale", ICPMode::RigidScale )
        .value( "AnyRigidXf", ICPMode::AnyRigidXf )
        .value( "OrthogonalAxis", ICPMode::OrthogonalAxis )
        .value( "FixedAxis", ICPMode::FixedAxis )
        .value( "TranslationOnly", ICPMode::TranslationOnly );

    emscripten::enum_<ICPExitType>( "ICPExitType" )
        .value( "NotStarted", ICPExitType::NotStarted )
        .value( "NotFoundSolution", ICPExitType::NotFoundSolution )
        .value( "MaxIterations", ICPExitType::MaxIterations )
        .value( "MaxBadIterations", ICPExitType::MaxBadIterations )
        .value( "StopMsdReached", ICPExitType::StopMsdReached );

    emscripten::class_<ICPProperties>( "ICPProperties" )
        .constructor<>()
        .property( "method", &ICPProperties::method )
        .property( "p2plAngleLimit", &ICPProperties::p2plAngleLimit )
        .property( "p2plScaleLimit", &ICPProperties::p2plScaleLimit )
        .property( "cosThreshold", &ICPProperties::cosThreshold )
        .property( "distThresholdSq", &ICPProperties::distThresholdSq )
        .property( "farDistFactor", &ICPProperties::farDistFactor )
        .property( "icpMode", &ICPProperties::icpMode )
        .property( "fixedRotationAxis", &ICPProperties::fixedRotationAxis )
        .property( "iterLimit", &ICPProperties::iterLimit )
        .property( "badIterStopCount", &ICPProperties::badIterStopCount )
        .property( "exitVal", &ICPProperties::exitVal )
        .property( "mutualClosest", &ICPProperties::mutualClosest )
        .property( "ignoreBdTgts", &ICPProperties::ignoreBdTgts );

    emscripten::class_<ICP>( "ICP" )
        .constructor<const MeshOrPointsXf&, const MeshOrPointsXf&, float>()
        .constructor<const MeshOrPointsXf&, const MeshOrPointsXf&, const VertBitSet&, const VertBitSet&>()
        .function( "setParams", &ICP::setParams )
        .function( "samplePoints", &ICP::samplePoints )
        .function( "autoSelectFloatXf", &ICP::autoSelectFloatXf )
        .function( "updatePointPairs", &ICP::updatePointPairs )
        .function( "getStatusInfo", &ICP::getStatusInfo )
        .function( "getNumSamples", &ICP::getNumSamples )
        .function( "getNumActivePairs", &ICP::getNumActivePairs )
        .function( "getMeanSqDistToPoint", &ICP::getMeanSqDistToPoint )
        .function( "getMeanSqDistToPlane", &ICP::getMeanSqDistToPlane )
        .function( "calculateTransformation", &ICP::calculateTransformation );
}
