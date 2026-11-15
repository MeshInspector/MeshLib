#include <MRCommonPlugins/Basic/MRFeatureAlignmentController.h>
#include <MRMesh/MRCircleObject.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRPlaneObject.h>
#include <MRMesh/MRChangeXfAction.h>
#include <MRMesh/MRChangeSceneAction.h>
#include <MRViewer/MRAppendHistory.h>
#include <MRViewer/MRHistoryStore.h>
#include <MRViewer/MRViewer.h>
#include <MRViewer/MRSceneReorder.h>
#include <gtest/gtest.h>
#include <optional>

namespace MR
{
namespace
{
using namespace FeatureAlignmentTool;
// Independent oracle: native hierarchy and full mesh copies, without catalog or controller state.
struct SceneSnapshot
{
    struct Node
    {
        ObjectPtr object;
        const Object *parent{};
        AffineXf3f local, world;
        std::vector<ObjectPtr> children;
        std::optional<Mesh> mesh;
    };
    std::vector<Node> nodes;
    explicit SceneSnapshot( const ObjectPtr &root )
    {
        const auto visit = [&]( const auto &self, const ObjectPtr &object ) -> void
        {
            Node node{ object, object->parent(), object->xf(), object->worldXf(), object->children(), {} };
            if ( const auto mesh = std::dynamic_pointer_cast<ObjectMesh>( object ); mesh && mesh->meshPtr() )
                node.mesh = *mesh->meshPtr();
            nodes.push_back( std::move( node ) );
            for ( const auto &child : object->children() )
                self( self, child );
        };
        visit( visit, root );
    }
    void expectUnchanged( const ObjectPtr &moving = {} ) const
    {
        for ( const auto &node : nodes )
        {
            SCOPED_TRACE( node.object->name() );
            EXPECT_EQ( node.object->parent(), node.parent );
            EXPECT_EQ( node.object->children(), node.children );
            if ( node.object != moving )
                EXPECT_EQ( node.object->xf(), node.local );
            bool movingDescendant = false;
            for ( auto *ancestor = node.object.get(); ancestor; ancestor = ancestor->parent() )
                movingDescendant = movingDescendant || ancestor == moving.get();
            if ( !movingDescendant )
                EXPECT_EQ( node.object->worldXf(), node.world );
            if ( node.mesh )
            {
                const auto object = std::dynamic_pointer_cast<ObjectMesh>( node.object );
                ASSERT_NE( object->meshPtr(), nullptr );
                EXPECT_TRUE( *node.mesh == *object->meshPtr() );
            }
        }
    }
};
class MRCommonPlugins : public testing::Test
{
protected:
    void SetUp() override
    {
        auto &viewer = Viewer::instanceRef();
        hadHistory_ = viewer.getGlobalHistoryStore() != nullptr;
        viewer.enableGlobalHistory( true );
        history_ = viewer.getGlobalHistoryStore();
        ASSERT_NE( history_, nullptr );
        history_->clear();
        root_ = std::make_shared<Object>();
        fixed_ = std::make_shared<ObjectMesh>();
        moving_ = std::make_shared<ObjectMesh>();
        unrelated_ = std::make_shared<ObjectMesh>();
        fixed_->setName( "Reference" );
        moving_->setName( "Floating" );
        unrelated_->setName( "Unrelated" );
        for ( const auto &part : { fixed_, moving_, unrelated_ } )
        {
            part->setMesh( std::make_shared<Mesh>( makeCube() ) );
            root_->addChild( part );
        }
        moving_->setXf( AffineXf3f::translation( { 3, 4, 20 } ) );
        auto child = std::make_shared<ObjectMesh>();
        child->setName( "Floating child" );
        child->setMesh( std::make_shared<Mesh>( makeCube() ) );
        child->setXf( AffineXf3f::translation( { 1, 2, 3 } ) );
        moving_->addChild( child );
        fixedPlane_ = std::make_shared<PlaneObject>();
        movingPlane_ = std::make_shared<PlaneObject>();
        fixed_->addChild( fixedPlane_ );
        moving_->addChild( movingPlane_ );
        selection_.fixed = fixed_;
        selection_.moving = moving_;
        selection_.fixedPrimary = fixedPlane_;
        selection_.movingPrimary = movingPlane_;
        settings_.normals = NormalRelation::Same;
        settings_.holeMode = HoleMode::Centers;
        controller_ = std::make_unique<Controller>( root_ );
        const auto configured = controller_->configure( selection_, settings_ );
        ASSERT_TRUE( configured ) << configured.error();
    }
    void TearDown() override
    {
        controller_.reset();
        Viewer::instanceRef().enableGlobalHistory( true );
        Viewer::instanceRef().getGlobalHistoryStore()->clear();
        if ( !hadHistory_ )
            Viewer::instanceRef().enableGlobalHistory( false );
    }
    void setHoles_( int count, bool mismatch = false )
    {
        for ( auto *locators : { &selection_.fixedLocators, &selection_.movingLocators } )
            for ( auto &feature : *locators )
            {
                if ( feature )
                    feature->detachFromParent();
                feature.reset();
            }
        for ( int i = 0; i < count; ++i )
        {
            auto fixed = std::make_shared<CircleObject>();
            auto moving = std::make_shared<CircleObject>();
            fixed->setRadius( 2 );
            moving->setRadius( 2 );
            fixed->setCenter( { i == 0 ? -10.f : 10.f, 0, i == 0 ? 4.f : -5.f } );
            moving->setCenter( { i == 0 ? -10.f : mismatch ? 12.f : 10.f, 0, i == 0 ? -9.f : 3.f } );
            fixed_->addChild( fixed );
            moving_->addChild( moving );
            selection_.fixedLocators[i] = fixed;
            selection_.movingLocators[i] = moving;
        }
    }
    void expectPlaneLock_() const
    {
        const auto fixedFrame = fixed_->worldXf(), movingFrame = moving_->worldXf();
        const auto fixedNormal = ( fixedFrame.A * fixedPlane_->getNormal() ).normalized();
        const auto movingNormal = ( movingFrame.A * movingPlane_->getNormal() ).normalized();
        EXPECT_NEAR( dot( movingNormal, fixedNormal ), int( settings_.normals ), 2e-5f );
        EXPECT_NEAR( dot( movingFrame( movingPlane_->getCenter() ) - fixedFrame( fixedPlane_->getCenter() ), fixedNormal ),
                     settings_.offset, 2e-4 );
    }
    void expectFrameNear_( const AffineXf3f &actual, const AffineXf3f &expected ) const
    {
        EXPECT_LT( ( actual.b - expected.b ).length(), 2e-4f );
        for ( int row = 0; row < 3; ++row )
            for ( int column = 0; column < 3; ++column )
                EXPECT_NEAR( actual.A[row][column], expected.A[row][column], 2e-5f );
    }
    ObjectPtr root_;
    std::shared_ptr<ObjectMesh> fixed_, moving_, unrelated_;
    std::shared_ptr<PlaneObject> fixedPlane_, movingPlane_;
    Selection selection_;
    Settings settings_;
    std::unique_ptr<Controller> controller_;
    HistoryStore *history_{};
    bool hadHistory_{};
};
} // namespace

TEST_F( MRCommonPlugins, FeatureAlignmentUnrelatedMetadataDoesNotInvalidatePreview )
{
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    unrelated_->setName( "Renamed unrelated part" );
    unrelated_->setXf( AffineXf3f::translation( { 7, 8, 9 } ) );
    EXPECT_FALSE( controller_->stalePreview() );
    EXPECT_TRUE( controller_->canApply() );
}
TEST_F( MRCommonPlugins, FeatureAlignmentNativeUndoCancelsOwnedVisualPreview )
{
    const auto before = moving_->xf();
    AppendHistory<ChangeXfAction>( "Unrelated edit", unrelated_, AffineXf3f::translation( { 7, 8, 9 } ) );
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    ASSERT_TRUE( controller_->hasPreview() );
    ASSERT_TRUE( history_->undo() );
    EXPECT_FALSE( controller_->hasPreview() );
    EXPECT_EQ( moving_->xf(), before );
    EXPECT_EQ( unrelated_->xf(), AffineXf3f{} );
}

TEST_F( MRCommonPlugins, FeatureAlignmentOneAndTwoHolesKeepPrimaryPlaneLock )
{
    for ( const auto relation : { NormalRelation::Same, NormalRelation::Opposing } )
        for ( int count : { 1, 2 } )
        {
            SCOPED_TRACE( count );
            SCOPED_TRACE( int( relation ) );
            setHoles_( count );
            settings_.normals = relation;
            settings_.offset = 3.25;
            const auto configured = controller_->configure( selection_, settings_ );
            ASSERT_TRUE( configured ) << configured.error();
            const SceneSnapshot before( root_ );
            const auto preview = controller_->preview();
            ASSERT_TRUE( preview ) << preview.error();
            EXPECT_TRUE( preview->acceptable() );
            EXPECT_EQ( preview->remainingDof, count == 1 ? 1 : 0 );
            expectPlaneLock_();
            before.expectUnchanged( moving_ );
            const auto canceled = controller_->invalidate();
            ASSERT_TRUE( canceled ) << canceled.error();
            before.expectUnchanged();
        }
}
TEST_F( MRCommonPlugins, FeatureAlignmentBalancingSharesTangentialMismatchWithoutChangingGap )
{
    for ( bool balance : { false, true } )
    {
        SCOPED_TRACE( balance );
        setHoles_( 2, true );
        settings_.balanceHoles = balance;
        settings_.linearTolerance = 3;
        settings_.offset = -2;
        const auto configured = controller_->configure( selection_, settings_ );
        ASSERT_TRUE( configured ) << configured.error();
        const SceneSnapshot before( root_ );
        const auto preview = controller_->preview();
        ASSERT_TRUE( preview ) << preview.error();
        ASSERT_EQ( preview->holes.size(), 2 );
        EXPECT_NEAR( preview->holes[0].position, balance ? 1. : 0., 2e-4 );
        EXPECT_NEAR( preview->holes[1].position, balance ? 1. : 2., 2e-4 );
        expectPlaneLock_();
        before.expectUnchanged( moving_ );
        const auto canceled = controller_->invalidate();
        ASSERT_TRUE( canceled ) << canceled.error();
        before.expectUnchanged();
    }
}
TEST_F( MRCommonPlugins, FeatureAlignmentRotatedNestedParentsPreserveProtectedGeometry )
{
    auto referenceGroup = std::make_shared<Object>(), movingGroup = std::make_shared<Object>();
    root_->addChild( referenceGroup );
    root_->addChild( movingGroup );
    referenceGroup->addChild( fixed_ );
    movingGroup->addChild( moving_ );
    root_->setXf( { Matrix3f::rotation( { 1, 2, 3 }, .4f ), { 120, 300, -50 } } );
    referenceGroup->setXf( { Matrix3f::rotation( { 2, -1, 4 }, .2f ), { 8, -4, 25 } } );
    movingGroup->setXf( { Matrix3f::rotation( { -1, 3, 1 }, -.3f ), { -7, 14, 6 } } );
    moving_->setXf( { Matrix3f::rotation( { 1, 0, 0 }, .7f ), { 30, 40, 50 } } );
    setHoles_( 2 );
    settings_.offset = 2.5;
    const auto configured = controller_->configure( selection_, settings_ );
    ASSERT_TRUE( configured ) << configured.error();
    const SceneSnapshot before( root_ );
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    EXPECT_TRUE( preview->acceptable() );
    expectPlaneLock_();
    before.expectUnchanged( moving_ );
    const auto applied = controller_->apply();
    ASSERT_TRUE( applied ) << applied.error();
    before.expectUnchanged( moving_ );
    ASSERT_TRUE( history_->undo() );
    before.expectUnchanged();
}
TEST_F( MRCommonPlugins, FeatureAlignmentVisualApplyRecordsPoseBeforePreview )
{
    const SceneSnapshot before( root_ );
    const auto original = moving_->xf();
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    const auto target = moving_->xf();
    ASSERT_NE( target, original );
    expectPlaneLock_();
    before.expectUnchanged( moving_ );
    ASSERT_TRUE( history_->getHistoryStack().empty() );
    const auto applied = controller_->apply();
    ASSERT_TRUE( applied ) << applied.error();
    EXPECT_FALSE( controller_->hasPreview() );
    EXPECT_EQ( moving_->xf(), target );
    ASSERT_EQ( history_->getHistoryStack().size(), 1 );
    ASSERT_TRUE( history_->undo() );
    before.expectUnchanged();
    ASSERT_TRUE( history_->redo() );
    EXPECT_EQ( moving_->xf(), target );
    before.expectUnchanged( moving_ );
}
TEST_F( MRCommonPlugins, FeatureAlignmentRepeatedVisualPreviewDoesNotAccumulateTransform )
{
    const SceneSnapshot before( root_ );
    auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    const auto target = moving_->xf();
    preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    EXPECT_EQ( moving_->xf(), target );
    EXPECT_TRUE( history_->getHistoryStack().empty() );
    const auto canceled = controller_->invalidate();
    ASSERT_TRUE( canceled ) << canceled.error();
    before.expectUnchanged();
}
TEST_F( MRCommonPlugins, FeatureAlignmentSettingsChangeCancelsOwnedPreview )
{
    const SceneSnapshot before( root_ );
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    settings_.offset = 5;
    const auto configured = controller_->configure( selection_, settings_ );
    ASSERT_TRUE( configured ) << configured.error();
    EXPECT_FALSE( controller_->hasPreview() );
    before.expectUnchanged();
}
TEST_F( MRCommonPlugins, FeatureAlignmentClosingCancelsOwnedPreview )
{
    const SceneSnapshot before( root_ );
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    controller_.reset();
    before.expectUnchanged();
    EXPECT_TRUE( history_->getHistoryStack().empty() );
}
TEST_F( MRCommonPlugins, FeatureAlignmentClosingPreservesNewerFloatingPose )
{
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    const auto newer = AffineXf3f::translation( { 7, 8, 9 } ) * moving_->xf();
    moving_->setXf( newer );
    const SceneSnapshot external( root_ );
    controller_.reset();
    external.expectUnchanged();
    EXPECT_EQ( moving_->xf(), newer );
}
TEST_F( MRCommonPlugins, FeatureAlignmentChangedPrimaryFeatureRejectsStaleApply )
{
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    fixedPlane_->setCenter( { 0, 0, 2 } );
    const SceneSnapshot external( root_ );
    EXPECT_TRUE( controller_->stalePreview() );
    EXPECT_FALSE( controller_->apply() );
    external.expectUnchanged();
    EXPECT_TRUE( history_->getHistoryStack().empty() );
}
TEST_F( MRCommonPlugins, FeatureAlignmentChangedReferenceRejectsStaleApply )
{
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    fixed_->setXf( AffineXf3f::translation( { 2, 0, 0 } ) );
    const SceneSnapshot external( root_ );
    EXPECT_TRUE( controller_->stalePreview() );
    EXPECT_FALSE( controller_->apply() );
    external.expectUnchanged();
}
TEST_F( MRCommonPlugins, FeatureAlignmentChangedFloatingRejectsStaleApplyWithoutOverwritingEdit )
{
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    moving_->setXf( AffineXf3f::translation( { 2, 0, 0 } ) * moving_->xf() );
    const SceneSnapshot external( root_ );
    EXPECT_FALSE( controller_->apply() );
    const auto released = controller_->invalidate( true );
    ASSERT_TRUE( released ) << released.error();
    external.expectUnchanged();
}
TEST_F( MRCommonPlugins, FeatureAlignmentSelectedMeshDirtyRejectsStaleApply )
{
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    moving_->varMesh()->points[VertId( 0 )].x += 2;
    moving_->setDirtyFlags( DIRTY_POSITION );
    const SceneSnapshot external( root_ );
    EXPECT_TRUE( controller_->stalePreview() );
    EXPECT_FALSE( controller_->apply() );
    external.expectUnchanged();
}
TEST_F( MRCommonPlugins, FeatureAlignmentUnrelatedMeshEditSurvivesApplyAndUndo )
{
    const auto original = moving_->xf();
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    unrelated_->varMesh()->points[VertId( 0 )].x += 2;
    unrelated_->setDirtyFlags( DIRTY_POSITION );
    const SceneSnapshot external( root_ );
    EXPECT_FALSE( controller_->stalePreview() );
    const auto applied = controller_->apply();
    ASSERT_TRUE( applied ) << applied.error();
    external.expectUnchanged( moving_ );
    ASSERT_TRUE( history_->undo() );
    EXPECT_EQ( moving_->xf(), original );
    external.expectUnchanged( moving_ );
}
TEST_F( MRCommonPlugins, FeatureAlignmentForeignFeatureMembershipRejectedAtPreview )
{
    unrelated_->addChild( movingPlane_ );
    const SceneSnapshot before( root_ );
    EXPECT_FALSE( controller_->preview() );
    before.expectUnchanged();
    EXPECT_TRUE( history_->getHistoryStack().empty() );
}
TEST_F( MRCommonPlugins, FeatureAlignmentOwnershipRecheckedImmediatelyBeforeApply )
{
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    unrelated_->addChild( movingPlane_ );
    const SceneSnapshot external( root_ );
    EXPECT_FALSE( controller_->apply() );
    external.expectUnchanged();
    EXPECT_TRUE( history_->getHistoryStack().empty() );
}
TEST_F( MRCommonPlugins, FeatureAlignmentExplicitSwapUsesNativeRolesWithoutSceneWrites )
{
    const SceneSnapshot before( root_ );
    const auto swapped = controller_->swapRoles();
    ASSERT_TRUE( swapped ) << swapped.error();
    EXPECT_EQ( controller_->selection().fixed, moving_ );
    EXPECT_EQ( controller_->selection().moving, fixed_ );
    before.expectUnchanged();
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    EXPECT_TRUE( preview->acceptable() );
    before.expectUnchanged( fixed_ );
    const auto canceled = controller_->invalidate();
    ASSERT_TRUE( canceled ) << canceled.error();
    before.expectUnchanged();
}
TEST_F( MRCommonPlugins, FeatureAlignmentSwapRejectsFloatingAncestorOfReference )
{
    fixed_->addChild( moving_ );
    const auto configured = controller_->configure( selection_, settings_ );
    ASSERT_TRUE( configured ) << configured.error();
    const SceneSnapshot before( root_ );
    EXPECT_FALSE( controller_->swapRoles() );
    before.expectUnchanged();
}
TEST_F( MRCommonPlugins, FeatureAlignmentMissingHistoryRejectsApplyWithoutWrite )
{
    const SceneSnapshot before( root_ );
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    const SceneSnapshot displayed( root_ );
    Viewer::instanceRef().enableGlobalHistory( false );
    EXPECT_FALSE( controller_->apply() );
    displayed.expectUnchanged();
    controller_.reset();
    before.expectUnchanged();
}

TEST_F( MRCommonPlugins, FeatureAlignmentNativeRedoCancelsOwnedVisualPreview )
{
    const auto before = moving_->xf();
    const auto unrelatedTarget = AffineXf3f::translation( { 7, 8, 9 } );
    AppendHistory<ChangeXfAction>( "Unrelated edit", unrelated_, unrelatedTarget );
    ASSERT_TRUE( history_->undo() );
    const auto preview = controller_->preview();
    ASSERT_TRUE( preview ) << preview.error();
    ASSERT_TRUE( history_->redo() );
    EXPECT_FALSE( controller_->hasPreview() );
    EXPECT_EQ( moving_->xf(), before );
    EXPECT_EQ( unrelated_->xf(), unrelatedTarget );
}
TEST_F( MRCommonPlugins, FeatureAlignmentViewOptionsFollowRolesAndExplicitRefresh )
{
    setHoles_( 2 );
    const auto configured = controller_->configure( selection_, settings_ );
    ASSERT_TRUE( configured ) << configured.error();
    const auto &view = controller_->viewState();
    EXPECT_TRUE( view.canPreview );
    EXPECT_FALSE( view.canApply );
    ASSERT_EQ( view.primaryFeatures[0].size(), 1 );
    ASSERT_EQ( view.primaryFeatures[1].size(), 1 );
    EXPECT_EQ( view.primaryFeatures[0][0].object, fixedPlane_ );
    EXPECT_EQ( view.primaryFeatures[1][0].object, movingPlane_ );
    EXPECT_EQ( view.locatorFeatures[0].size(), 2 );
    EXPECT_EQ( view.locatorFeatures[1].size(), 2 );
    EXPECT_EQ( view.clockingFeatures[0].size(), 3 );
    EXPECT_EQ( view.clockingFeatures[1].size(), 3 );
    auto added = std::make_shared<PlaneObject>();
    fixed_->addChild( added );
    controller_->refresh();
    EXPECT_EQ( controller_->viewState().primaryFeatures[0].size(), 2 );
    const auto swapped = controller_->swapRoles();
    ASSERT_TRUE( swapped ) << swapped.error();
    const auto &reversed = controller_->viewState();
    ASSERT_EQ( reversed.primaryFeatures[0].size(), 1 );
    EXPECT_EQ( reversed.primaryFeatures[0][0].object, movingPlane_ );
    EXPECT_EQ( reversed.primaryFeatures[1].size(), 2 );
}
TEST_F( MRCommonPlugins, FeatureAlignmentObserversStaySelectedAndDoNotAccumulate )
{
    const auto movingModels = moving_->meshChangedSignal.num_slots();
    const auto fixedModels = fixed_->meshChangedSignal.num_slots();
    const auto movingPoses = moving_->worldXfChangedSignal.num_slots();
    const auto fixedPoses = fixed_->worldXfChangedSignal.num_slots();
    EXPECT_GT( movingModels, 0 );
    EXPECT_GT( fixedModels, 0 );
    EXPECT_GT( movingPoses, 0 );
    EXPECT_GT( fixedPoses, 0 );
    for ( int repeat = 0; repeat < 5; ++repeat )
    {
        settings_.offset += .1;
        const auto configured = controller_->configure( selection_, settings_ );
        ASSERT_TRUE( configured ) << configured.error();
        EXPECT_EQ( moving_->meshChangedSignal.num_slots(), movingModels );
        EXPECT_EQ( fixed_->meshChangedSignal.num_slots(), fixedModels );
        EXPECT_EQ( moving_->worldXfChangedSignal.num_slots(), movingPoses );
        EXPECT_EQ( fixed_->worldXfChangedSignal.num_slots(), fixedPoses );
        EXPECT_EQ( unrelated_->meshChangedSignal.num_slots(), 0 );
        EXPECT_EQ( unrelated_->worldXfChangedSignal.num_slots(), 0 );
    }
    controller_.reset();
    EXPECT_EQ( moving_->meshChangedSignal.num_slots(), 0 );
    EXPECT_EQ( fixed_->meshChangedSignal.num_slots(), 0 );
    EXPECT_EQ( moving_->worldXfChangedSignal.num_slots(), 0 );
    EXPECT_EQ( fixed_->worldXfChangedSignal.num_slots(), 0 );
}
TEST_F( MRCommonPlugins, FeatureAlignmentRolesUseSuppliedSelectionAndRecognizedTreeOrder )
{
    fixed_->select( false );
    moving_->select( false );
    unrelated_->select( true );
    const SceneSnapshot before( root_ );
    const std::vector<std::vector<std::shared_ptr<const Object>>> valid{
        { moving_, fixed_ }, { fixed_, moving_ }, { movingPlane_, fixedPlane_ },
        { fixed_, fixedPlane_, moving_ }
    };
    for ( const auto &selected : valid )
    {
        const auto roles = selectedHierarchyRoles( root_, selected );
        ASSERT_TRUE( roles ) << roles.error();
        EXPECT_EQ( roles->fixed, fixed_ );
        EXPECT_EQ( roles->moving, moving_ );
    }
    auto foreignRoot = std::make_shared<Object>();
    auto foreign = std::make_shared<ObjectMesh>();
    foreign->setMesh( std::make_shared<Mesh>( makeCube() ) );
    foreignRoot->addChild( foreign );
    auto unrecognized = std::make_shared<ObjectMesh>();
    unrecognized->setMesh( std::make_shared<Mesh>( makeCube() ) );
    root_->addChild( unrecognized, false );
    const std::vector<std::vector<std::shared_ptr<const Object>>> invalid{
        {}, { fixed_ }, { fixed_, moving_, unrelated_ }, { fixed_, fixedPlane_ },
        { moving_, foreign }, { fixed_, unrecognized }
    };
    for ( const auto &selected : invalid )
        EXPECT_FALSE( selectedHierarchyRoles( root_, selected ) );
    before.expectUnchanged();
}
TEST_F( MRCommonPlugins, FeatureAlignmentNativeReorderIntoPreviewPreservesIncomingObject )
{
    const auto before = moving_->xf();
    const auto incomingWorld = unrelated_->worldXf();
    const auto referenceWorld = fixed_->worldXf();
    const Mesh incomingMesh = *unrelated_->meshPtr();
    ASSERT_TRUE( controller_->preview() );
    ASSERT_TRUE( sceneReorderWithUndo( { { unrelated_.get() }, moving_.get() } ) );
    EXPECT_EQ( moving_->xf(), before );
    EXPECT_FALSE( controller_->hasPreview() );
    EXPECT_EQ( unrelated_->worldXf(), incomingWorld );
    EXPECT_EQ( fixed_->worldXf(), referenceWorld );
    ASSERT_TRUE( history_->undo() );
    EXPECT_EQ( unrelated_->worldXf(), incomingWorld );
    ASSERT_TRUE( history_->redo() );
    EXPECT_EQ( unrelated_->worldXf(), incomingWorld );
    EXPECT_EQ( fixed_->worldXf(), referenceWorld );
    EXPECT_TRUE( *unrelated_->meshPtr() == incomingMesh );
}
TEST_F( MRCommonPlugins, FeatureAlignmentNativeReorderPreservesReferenceFeature )
{
    const auto before = moving_->xf();
    const auto featureWorld = fixedPlane_->worldXf();
    const auto referenceWorld = fixed_->worldXf();
    ASSERT_TRUE( controller_->preview() );
    ASSERT_TRUE( sceneReorderWithUndo( { { fixedPlane_.get() }, moving_.get() } ) );
    EXPECT_EQ( moving_->xf(), before );
    EXPECT_EQ( fixedPlane_->worldXf(), featureWorld );
    EXPECT_EQ( fixed_->worldXf(), referenceWorld );
    ASSERT_TRUE( history_->undo() );
    EXPECT_EQ( fixedPlane_->parent(), fixed_.get() );
    EXPECT_EQ( fixedPlane_->worldXf(), featureWorld );
    ASSERT_TRUE( history_->redo() );
    EXPECT_EQ( fixedPlane_->worldXf(), featureWorld );
}
TEST_F( MRCommonPlugins, FeatureAlignmentNativeReorderPreservesIncomingReferencePart )
{
    const auto before = moving_->xf();
    const auto referenceWorld = fixed_->worldXf();
    const auto featureWorld = fixedPlane_->worldXf();
    const Mesh referenceMesh = *fixed_->meshPtr();
    ASSERT_TRUE( controller_->preview() );
    ASSERT_TRUE( sceneReorderWithUndo( { { fixed_.get() }, moving_.get() } ) );
    EXPECT_EQ( moving_->xf(), before );
    EXPECT_FALSE( controller_->hasPreview() );
    EXPECT_EQ( fixed_->worldXf(), referenceWorld );
    EXPECT_EQ( fixedPlane_->worldXf(), featureWorld );
    ASSERT_TRUE( history_->undo() );
    EXPECT_EQ( fixed_->worldXf(), referenceWorld );
    ASSERT_TRUE( history_->redo() );
    EXPECT_EQ( fixed_->worldXf(), referenceWorld );
    EXPECT_EQ( fixedPlane_->worldXf(), featureWorld );
    EXPECT_TRUE( *fixed_->meshPtr() == referenceMesh );
}
TEST_F( MRCommonPlugins, FeatureAlignmentNativeReorderPreservesMultipleRotatedBranches )
{
    root_->setXf( { Matrix3f::rotation( { 1, 2, 3 }, .3f ), { 100, -30, 20 } } );
    moving_->setXf( { Matrix3f::rotation( { 2, 1, 3 }, .4f ), { 3, 4, 20 } } );
    unrelated_->setXf( { Matrix3f::rotation( { 1, 3, 2 }, .2f ), { 7, 8, 9 } } );
    auto group = std::make_shared<Object>();
    auto child = std::make_shared<ObjectMesh>();
    child->setMesh( std::make_shared<Mesh>( makeCube() ) );
    group->setXf( { Matrix3f::rotation( { 3, 1, 2 }, -.2f ), { -6, 12, 3 } } );
    child->setXf( AffineXf3f::translation( { 4, -7, 9 } ) );
    root_->addChild( group );
    group->addChild( child );
    const auto before = moving_->xf();
    const auto unrelatedWorld = unrelated_->worldXf();
    const auto groupWorld = group->worldXf();
    const auto childWorld = child->worldXf();
    const auto referenceWorld = fixed_->worldXf();
    const Mesh childMesh = *child->meshPtr();
    ASSERT_TRUE( controller_->preview() );
    ASSERT_TRUE( sceneReorderWithUndo( { { unrelated_.get(), group.get() }, moving_.get() } ) );
    EXPECT_EQ( moving_->xf(), before );
    EXPECT_EQ( fixed_->worldXf(), referenceWorld );
    for ( int step = 0; step < 3; ++step )
    {
        expectFrameNear_( unrelated_->worldXf(), unrelatedWorld );
        expectFrameNear_( group->worldXf(), groupWorld );
        expectFrameNear_( child->worldXf(), childWorld );
        if ( step == 0 )
            ASSERT_TRUE( history_->undo() );
        else if ( step == 1 )
            ASSERT_TRUE( history_->redo() );
    }
    EXPECT_TRUE( *child->meshPtr() == childMesh );
}
TEST_F( MRCommonPlugins, FeatureAlignmentNativeReorderOutOfPreviewPreservesUserPlacement )
{
    const auto before = moving_->xf();
    const auto originalFeatureWorld = movingPlane_->worldXf();
    ASSERT_TRUE( controller_->preview() );
    const auto previewFeatureWorld = movingPlane_->worldXf();
    ASSERT_TRUE( sceneReorderWithUndo( { { movingPlane_.get() }, unrelated_.get() } ) );
    EXPECT_EQ( moving_->xf(), before );
    EXPECT_EQ( movingPlane_->worldXf(), previewFeatureWorld );
    ASSERT_TRUE( history_->undo() );
    EXPECT_EQ( movingPlane_->parent(), moving_.get() );
    EXPECT_EQ( movingPlane_->worldXf(), originalFeatureWorld );
    ASSERT_TRUE( history_->redo() );
    EXPECT_EQ( movingPlane_->worldXf(), previewFeatureWorld );
}
TEST_F( MRCommonPlugins, FeatureAlignmentNativeReorderWithEqualParentFramesRetainsUndo )
{
    moving_->setXf( AffineXf3f::translation( { 0, 0, 20 } ) );
    const auto before = moving_->xf();
    const auto incomingWorld = unrelated_->worldXf();
    ASSERT_TRUE( controller_->preview() );
    ASSERT_EQ( moving_->worldXf(), root_->worldXf() );
    ASSERT_TRUE( sceneReorderWithUndo( { { unrelated_.get() }, moving_.get() } ) );
    EXPECT_EQ( moving_->xf(), before );
    EXPECT_EQ( unrelated_->worldXf(), incomingWorld );
    EXPECT_EQ( history_->getHistoryStack().size(), 1 );
    ASSERT_TRUE( history_->undo() );
    EXPECT_EQ( unrelated_->parent(), root_.get() );
    EXPECT_EQ( unrelated_->worldXf(), incomingWorld );
    ASSERT_TRUE( history_->redo() );
    EXPECT_EQ( unrelated_->worldXf(), incomingWorld );
}
TEST_F( MRCommonPlugins, FeatureAlignmentNativeReorderReferenceWithEqualParentFramesRetainsUndo )
{
    moving_->setXf( AffineXf3f::translation( { 0, 0, 20 } ) );
    const auto before = moving_->xf();
    const auto referenceWorld = fixed_->worldXf();
    const auto featureWorld = fixedPlane_->worldXf();
    ASSERT_TRUE( controller_->preview() );
    ASSERT_EQ( moving_->worldXf(), root_->worldXf() );
    ASSERT_TRUE( sceneReorderWithUndo( { { fixed_.get() }, moving_.get() } ) );
    EXPECT_EQ( moving_->xf(), before );
    EXPECT_EQ( fixed_->worldXf(), referenceWorld );
    EXPECT_EQ( fixedPlane_->worldXf(), featureWorld );
    ASSERT_TRUE( history_->undo() );
    EXPECT_EQ( fixed_->worldXf(), referenceWorld );
    ASSERT_TRUE( history_->redo() );
    EXPECT_EQ( fixed_->worldXf(), referenceWorld );
    EXPECT_EQ( fixedPlane_->worldXf(), featureWorld );
}
TEST_F( MRCommonPlugins, FeatureAlignmentNativeReorderMixedParentFramesRetainsUndo )
{
    moving_->setXf( AffineXf3f::translation( { 0, 0, 20 } ) );
    auto otherParent = std::make_shared<Object>();
    auto other = std::make_shared<Object>();
    root_->addChild( otherParent );
    otherParent->setXf( AffineXf3f::translation( { 0, 0, 7 } ) );
    otherParent->addChild( other );
    other->setXf( AffineXf3f::translation( { 0, 0, 2 } ) );
    const auto incomingWorld = unrelated_->worldXf();
    const auto otherWorld = other->worldXf();
    ASSERT_TRUE( controller_->preview() );
    ASSERT_EQ( moving_->worldXf(), root_->worldXf() );
    ASSERT_TRUE( sceneReorderWithUndo( { { unrelated_.get(), other.get() }, moving_.get() } ) );
    for ( int step = 0; step < 3; ++step )
    {
        EXPECT_EQ( unrelated_->worldXf(), incomingWorld );
        EXPECT_EQ( other->worldXf(), otherWorld );
        if ( step == 0 )
            ASSERT_TRUE( history_->undo() );
        else if ( step == 1 )
            ASSERT_TRUE( history_->redo() );
    }
}
TEST_F( MRCommonPlugins, FeatureAlignmentNativeCreationDuringPreviewRetainsSingleUndo )
{
    for ( const bool nestedScope : { false, true } )
    {
        SCOPED_TRACE( nestedScope );
        const SceneSnapshot before( root_ );
        const auto floatingBefore = moving_->xf();
        auto added = std::make_shared<ObjectMesh>();
        added->setMesh( std::make_shared<Mesh>( makeCube() ) );
        const Mesh addedMesh = *added->meshPtr();
        const AffineXf3f placement{ Matrix3f::rotation( { 1, 2, 3 }, .25f ), { 8, 9, 10 } };
        ASSERT_TRUE( controller_->preview() );
        const auto create = [&]
        {
            const auto action = std::make_shared<ChangeSceneAction>( "Create object", added, ChangeSceneAction::Type::AddObject );
            moving_->addChild( added );
            added->setWorldXf( placement );
            AppendHistory( action );
        };
        if ( nestedScope )
        {
            ScopeHistory outer( "Create group" );
            ScopeHistory inner( "Create object" );
            create();
        }
        else
            create();
        EXPECT_EQ( moving_->xf(), floatingBefore );
        EXPECT_FALSE( controller_->hasPreview() );
        expectFrameNear_( added->worldXf(), placement );
        EXPECT_EQ( history_->getHistoryStack().size(), 1 );
        ASSERT_TRUE( history_->undo() );
        EXPECT_EQ( added->parent(), nullptr );
        before.expectUnchanged();
        ASSERT_TRUE( history_->redo() );
        expectFrameNear_( added->worldXf(), placement );
        EXPECT_TRUE( *added->meshPtr() == addedMesh );
        ASSERT_TRUE( history_->undo() );
        before.expectUnchanged();
        history_->clear();
    }
}
} // namespace MR
