#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRRibbonRegisterItem.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRViewerInstance.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRCombinedHistoryAction.h"
#include "MRMesh/MRPositionedText.h"
#include "MRSymbolMesh/MRObjectLabel.h"

namespace MR
{

class RotatorPlugin : public StateListenerPlugin<PreDrawListener>
{
public:
    RotatorPlugin();

    void drawDialog( float menuScaling, ImGuiContext* ) override;
    bool blocking() const override { return false; }

private:
    bool onEnable_() override;
    bool onDisable_() override;
    void preDraw_() override;
    /// returns true only if on the top of undo stack there is not our own action (not to make new undo on each frame)
    bool shouldCreateNewHistoryAction_( const std::vector<std::shared_ptr<Object>>& selObjs ) const;

    float rotationSpeed_ = 5 * PI_F / 180;
    bool rotateCamera_ = true;

    std::weak_ptr<CombinedHistoryAction> myLastHistoryAction_;
    std::shared_ptr<MR::ObjectLabel> label_;
};

RotatorPlugin::RotatorPlugin() :
    StateListenerPlugin( "Rotator" )
{
}

void RotatorPlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 150.0f * menuScaling;
    if ( !ImGuiBeginWindow_( { .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    ImGui::SetNextItemWidth( 90.0f * menuScaling );
    UI::drag<AngleUnit>( "Speed", rotationSpeed_, 0.01f, -2 * PI_F, 2 * PI_F );
    UI::setTooltipIfHovered( "The speed of camera rotation in degrees per second. The sign of this value specifies the direction of rotation.", menuScaling );

    UI::checkbox( "Rotate Camera", &rotateCamera_ ); 
    UI::setTooltipIfHovered( "If selected then camera is rotated around scene's center. Otherwise selected objects are rotated, each around its center.", menuScaling );

    ImGui::EndCustomStatePlugin();
}

bool RotatorPlugin::onEnable_()
{
    return true;
}

bool RotatorPlugin::onDisable_()
{
    myLastHistoryAction_.reset();
    return true;
}

bool RotatorPlugin::shouldCreateNewHistoryAction_( const std::vector<std::shared_ptr<Object>>& selObjs ) const
{
    if ( selObjs.empty() )
        return false;
    const auto& hs = HistoryStore::getViewerInstance();
    if ( !hs )
        return false;
    auto a = myLastHistoryAction_.lock();
    if ( !a )
        return true;
    if ( hs->getLastAction( HistoryAction::Type::Undo ) != a )
        return true;
    if ( a->getStack().size() != selObjs.size() )
        return true;
    for ( int i = 0; i < selObjs.size(); ++i )
    {
        auto c = dynamic_cast<ChangeXfAction*>( a->getStack()[i].get() );
        assert( c );
        if ( !c || c->obj() != selObjs[i] )
            return true;
    }
    return false;
}

MR::AffineXf3f worldToBasis(
    const MR::Vector3f& directionMapToY, const MR::Vector3f& directionMapToZ, const MR::Vector3f& basisOrigin )
{
    const auto directionMapToX = MR::cross( directionMapToY, directionMapToZ );
    MR::Matrix3f rotation;
    rotation[0] = directionMapToX;
    rotation[1] = directionMapToY;
    rotation[2] = directionMapToZ;
    rotation = rotation.transposed();
    return { rotation, basisOrigin };
}

void RotatorPlugin::preDraw_()
{
    if ( !label_ )
    {
        label_ = std::make_shared<MR::ObjectLabel>();
        PositionedText txtOcclusal;
        txtOcclusal.position = MR::Vector3f{ 0, .5f, 0 };
        txtOcclusal.text = "One";
        label_->setLabel( txtOcclusal );
        label_->setVisualizeProperty( false, MR::VisualizeMaskType::DepthTest, MR::ViewportMask::all() );
        label_->setVisualizeProperty(
            true, MR::LabelVisualizePropertyType::Contour, MR::ViewportMask::all() );
        label_->setAncillary( true );
        label_->setPickable( false );
        label_->setVisible( true );
        MR::SceneRoot::get().addChild( label_ );
    }

    auto& viewport1 = Viewport::get( ViewportId( 1 ) );
    auto& viewport2 = Viewport::get( ViewportId( 2 ) );

    viewport2.setCameraTrackballAngle( viewport1.getParameters().cameraTrackballAngle );

    if ( label_ )
    {
        auto up = viewport1.getUpDirection();
        auto back = viewport1.getBackwardDirection();
        auto cameraTranslation = viewport1.getParameters().cameraTranslation;

        auto moveXf = worldToBasis( back, up, {} );
        moveXf.b -= cameraTranslation;
        label_->setXf( moveXf );

        viewport2.cameraLookAlong( back, up );
        viewport2.setCameraTranslation( cameraTranslation );
    }
/*
    auto & viewport = Viewport::get();
    Vector3f sceneCenter;
    if ( auto sceneBox = viewport.getSceneBox(); sceneBox.valid() )
        sceneCenter = sceneBox.center();

    const auto deltaAngle = ImGui::GetIO().DeltaTime * rotationSpeed_;

    if ( rotateCamera_ )
    {
        viewport.cameraRotateAround( Line3f{ sceneCenter, viewport.getUpDirection() }, -deltaAngle );
    }
    else
    {
        const auto selObjs = getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        bool appendHistory = shouldCreateNewHistoryAction_( selObjs );
        std::optional<ScopeHistory> scope;
        if ( appendHistory )
        {
            scope.emplace( "Rotator" );
            myLastHistoryAction_ = scope->combinedAction();
        }
        const auto rotMat = Matrix3f::rotation( viewport.getUpDirection(), deltaAngle );
        for ( const auto & obj : selObjs )
        {
            const auto wXf = obj->worldXf();
            const auto wCenter = obj->getWorldTreeBox().center();
            if ( appendHistory )
                AppendHistory<ChangeXfAction>( obj->name(), obj );
            obj->setWorldXf( AffineXf3f::xfAround( rotMat, wCenter ) * wXf );
        }
    }

    incrementForceRedrawFrames();*/
}

MR_REGISTER_RIBBON_ITEM( RotatorPlugin )

}
