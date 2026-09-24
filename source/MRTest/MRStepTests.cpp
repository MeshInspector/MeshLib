#include <MRIOExtras/config.h>
#if !defined __EMSCRIPTEN__ && !defined MRIOEXTRAS_NO_STEP
#include <MRIOExtras/MRStep.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRObject.h>
#include <MRMesh/MRObjectMesh.h>
#include <gtest/gtest.h>
#include <sstream>

namespace MR
{

namespace
{

/// a STEP model holding a single body: one triangular face making up one shell
const std::string cOneBodyStep = R"STEP(ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Open CASCADE Model'),'2;1');
FILE_NAME('Open CASCADE Shape Model','2026-09-24T14:58:15',('Author'),(
    'Open CASCADE'),'Open CASCADE STEP processor 7.5','Open CASCADE 7.5'
  ,'Unknown');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));
ENDSEC;
DATA;
#1 = APPLICATION_PROTOCOL_DEFINITION('international standard',
  'automotive_design',2000,#2);
#2 = APPLICATION_CONTEXT(
  'core data for automotive mechanical design processes');
#3 = SHAPE_DEFINITION_REPRESENTATION(#4,#10);
#4 = PRODUCT_DEFINITION_SHAPE('','',#5);
#5 = PRODUCT_DEFINITION('design','',#6,#9);
#6 = PRODUCT_DEFINITION_FORMATION('','',#7);
#7 = PRODUCT('Open CASCADE STEP translator 7.5 1',
  'Open CASCADE STEP translator 7.5 1','',(#8));
#8 = PRODUCT_CONTEXT('',#2,'mechanical');
#9 = PRODUCT_DEFINITION_CONTEXT('part definition',#2,'design');
#10 = MANIFOLD_SURFACE_SHAPE_REPRESENTATION('',(#11,#15),#73);
#11 = AXIS2_PLACEMENT_3D('',#12,#13,#14);
#12 = CARTESIAN_POINT('',(0.,0.,0.));
#13 = DIRECTION('',(0.,0.,1.));
#14 = DIRECTION('',(1.,0.,-0.));
#15 = SHELL_BASED_SURFACE_MODEL('',(#16));
#16 = OPEN_SHELL('',(#17));
#17 = ADVANCED_FACE('',(#18),#32,.T.);
#18 = FACE_BOUND('',#19,.T.);
#19 = EDGE_LOOP('',(#20,#43,#59));
#20 = ORIENTED_EDGE('',*,*,#21,.T.);
#21 = EDGE_CURVE('',#22,#24,#26,.T.);
#22 = VERTEX_POINT('',#23);
#23 = CARTESIAN_POINT('',(0.,0.,0.));
#24 = VERTEX_POINT('',#25);
#25 = CARTESIAN_POINT('',(10.,0.,0.));
#26 = SURFACE_CURVE('',#27,(#31),.PCURVE_S1.);
#27 = LINE('',#28,#29);
#28 = CARTESIAN_POINT('',(0.,0.,0.));
#29 = VECTOR('',#30,1.);
#30 = DIRECTION('',(1.,0.,0.));
#31 = PCURVE('',#32,#37);
#32 = PLANE('',#33);
#33 = AXIS2_PLACEMENT_3D('',#34,#35,#36);
#34 = CARTESIAN_POINT('',(3.535533905933,3.535533905933,0.));
#35 = DIRECTION('',(0.,0.,1.));
#36 = DIRECTION('',(1.,0.,-0.));
#37 = DEFINITIONAL_REPRESENTATION('',(#38),#42);
#38 = LINE('',#39,#40);
#39 = CARTESIAN_POINT('',(-3.535533905933,-3.535533905933));
#40 = VECTOR('',#41,1.);
#41 = DIRECTION('',(1.,0.));
#42 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2) 
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE',''
  ) );
#43 = ORIENTED_EDGE('',*,*,#44,.T.);
#44 = EDGE_CURVE('',#24,#45,#47,.T.);
#45 = VERTEX_POINT('',#46);
#46 = CARTESIAN_POINT('',(0.,10.,0.));
#47 = SURFACE_CURVE('',#48,(#52),.PCURVE_S1.);
#48 = LINE('',#49,#50);
#49 = CARTESIAN_POINT('',(10.,0.,0.));
#50 = VECTOR('',#51,1.);
#51 = DIRECTION('',(-0.707106781187,0.707106781187,0.));
#52 = PCURVE('',#32,#53);
#53 = DEFINITIONAL_REPRESENTATION('',(#54),#58);
#54 = LINE('',#55,#56);
#55 = CARTESIAN_POINT('',(6.464466094067,-3.535533905933));
#56 = VECTOR('',#57,1.);
#57 = DIRECTION('',(-0.707106781187,0.707106781187));
#58 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2) 
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE',''
  ) );
#59 = ORIENTED_EDGE('',*,*,#60,.T.);
#60 = EDGE_CURVE('',#45,#22,#61,.T.);
#61 = SURFACE_CURVE('',#62,(#66),.PCURVE_S1.);
#62 = LINE('',#63,#64);
#63 = CARTESIAN_POINT('',(0.,10.,0.));
#64 = VECTOR('',#65,1.);
#65 = DIRECTION('',(0.,-1.,0.));
#66 = PCURVE('',#32,#67);
#67 = DEFINITIONAL_REPRESENTATION('',(#68),#72);
#68 = LINE('',#69,#70);
#69 = CARTESIAN_POINT('',(-3.535533905933,6.464466094067));
#70 = VECTOR('',#71,1.);
#71 = DIRECTION('',(0.,-1.));
#72 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2) 
PARAMETRIC_REPRESENTATION_CONTEXT() REPRESENTATION_CONTEXT('2D SPACE',''
  ) );
#73 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) 
GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#77)) GLOBAL_UNIT_ASSIGNED_CONTEXT(
(#74,#75,#76)) REPRESENTATION_CONTEXT('Context #1',
  '3D Context with UNIT and UNCERTAINTY') );
#74 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
#75 = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );
#76 = ( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() );
#77 = UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#74,
  'distance_accuracy_value','confusion accuracy');
#78 = PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#7));
ENDSEC;
END-ISO-10303-21;
)STEP";

std::shared_ptr<Object> loadOneBody( bool forceLoadSubShapes )
{
    MeshLoad::StepLoadSettings stepSettings;
    stepSettings.forceLoadSubShapes = forceLoadSubShapes;

    std::istringstream in( cOneBodyStep );
    auto res = MeshLoad::fromSceneStepFile( in, MeshLoadSettings {}, stepSettings );
    EXPECT_TRUE( res.has_value() );
    if ( !res )
        return {};
    return *res;
}

} // namespace

TEST( MRMesh, StepSingleBodyLoadedAsMesh )
{
    // by default a single-body shape loads as one mesh object, without an extra group level
    auto scene = loadOneBody( false );
    ASSERT_TRUE( scene );
    ASSERT_EQ( scene->children().size(), 1 );

    auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( scene->children().front() );
    ASSERT_TRUE( objMesh );
    ASSERT_TRUE( objMesh->meshPtr() );
    EXPECT_EQ( objMesh->mesh()->topology.numValidFaces(), 1 );
    EXPECT_TRUE( objMesh->children().empty() );
}

TEST( MRMesh, StepForceLoadSubShapes )
{
    // with the option on, the same shape is split into per-body children
    auto scene = loadOneBody( true );
    ASSERT_TRUE( scene );
    ASSERT_EQ( scene->children().size(), 1 );

    const auto& group = scene->children().front();
    EXPECT_FALSE( std::dynamic_pointer_cast<ObjectMesh>( group ) );
    ASSERT_EQ( group->children().size(), 1 );

    auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( group->children().front() );
    ASSERT_TRUE( objMesh );
    ASSERT_TRUE( objMesh->meshPtr() );
    // the geometry is the same, only the scene structure changes
    EXPECT_EQ( objMesh->mesh()->topology.numValidFaces(), 1 );
}

} // namespace MR
#endif
